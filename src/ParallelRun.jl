module ParallelRun

using Base.Threads
using Base.Dates

import FITSIO
import JLD
import WCS

import ..Celeste: detect_sources
import ..Config
import ..Log
using ..Model
import ..SDSSIO
import ..SDSSIO: RunCamcolField, IOStrategy, PlainFITSStrategy
import ..PSF
import ..SEP
import ..Coordinates: angular_separation, match_coordinates
import ..MCMC

export BoundingBox

include("joint_infer.jl")

abstract type ParallelismStrategy; end
struct ThreadsStrategy <: ParallelismStrategy; end


# In production mode, rather the development mode, always catch exceptions
const is_production_run = haskey(ENV, "CELESTE_PROD") && ENV["CELESTE_PROD"] != ""

# ------ bounding box ------
struct BoundingBox
    ramin::Float64
    ramax::Float64
    decmin::Float64
    decmax::Float64
end

function BoundingBox(ramin::String, ramax::String, decmin::String, decmax::String)
    BoundingBox(parse(Float64, ramin),
                parse(Float64, ramax),
                parse(Float64, decmin),
                parse(Float64, decmax))
end


# ------
# optimization

# optimization result container
struct OptimizedSource
    init_ra::Float64
    init_dec::Float64
    vs::Vector{Float64}
    is_sky_bad::Bool
end


# The sky intensity does not appear to match the true background intensity
# for some light sources. This function flags light sources whose infered
# brightness may be inaccurate because the background intensity estimates
# are off.
function bad_sky(ce, images)
    # The 'i' band is a pretty bright one, so I used it here.
    img_index = findfirst(img -> img.b == 4, images)
    if img_index < 0
        return false
    end

    img = images[img_index]

    # Get sky at location of the object
    pixel_center = WCS.world_to_pix(img.wcs, ce.pos)
    h = max(1, min(round(Int, pixel_center[1]), size(img.pixels, 1)))
    w = max(1, min(round(Int, pixel_center[2]), size(img.pixels, 2)))

    claimed_sky = img.sky[h, w] * img.nelec_per_nmgy[h]

    # Determine background in a 50-pixel radius box around the object
    box = Model.box_around_point(img.wcs, ce.pos, 50.0)
    h_range, w_range = Model.clamp_box(box, (img.H, img.W))
    observed_sky = median(filter(!isnan, img.pixels[h_range, w_range]))

    # A 5 photon-per-pixel disparity can really add up if the light sources
    # covers a lot of pixels.
    return (claimed_sky + 5) < observed_sky
end


"""
Optimize the `ts`th element of `target_sources`.
Used only for one_node_single_infer, not one_node_joint_infer.
"""
function process_source(config::Config,
                        ts::Int,
                        catalog::Vector{CatalogEntry},
                        patches::Matrix{ImagePatch},
                        target_sources::Vector{Int},
                        neighbor_map::Vector{Vector{Int}},
                        images::Vector{<:Image})
    neighbors = catalog[neighbor_map[ts]]
    if length(neighbors) > 100
        msg = string("object at RA, Dec = $(entry.pos) has an excessive",
                     "number ($(length(neighbors))) of neighbors")
        Log.warn(msg)
    end

    s = target_sources[ts]
    entry = catalog[s]
    cat_local = vcat([entry], neighbors)

    # Limit patches to just the active source and its neighbors.
    idxs = vcat([s], neighbor_map[ts])
    patches = patches[idxs, :]

    vp = DeterministicVI.init_sources([1], cat_local)
    ea = DeterministicVI.ElboArgs(images, patches, [1])

    tic()
    f_evals, max_f, max_x, nm_result = DeterministicVI.ElboMaximize.maximize!(ea, vp)
    Log.info("#$(ts) at ($(entry.pos[1]), $(entry.pos[2])): $(toq()) secs")

    vs_opt = vp[1]
    is_sky_bad = bad_sky(entry, images)
    return OptimizedSource(entry.pos[1], entry.pos[2], vs_opt, is_sky_bad)
end


"""
Run MCMC to process a source.  Returns
"""
function process_source_mcmc(config::Config,
                             ts::Int,
                             catalog::Vector{CatalogEntry},
                             patches::Matrix{ImagePatch},
                             target_sources::Vector{Int},
                             neighbor_map::Vector{Vector{Int}},
                             images::Vector{<:Image};
                             use_ais::Bool=true)
    # subselect source, select active source and neighbor set
    s = target_sources[ts]
    entry = catalog[s]
    neighbors = catalog[neighbor_map[ts]]
    if length(neighbors) > 100
        msg = string("objid $(entry.objid) [ra: $(entry.pos)] has an excessive",
                     "number ($(length(neighbors))) of neighbors")
        Log.warn(msg)
    end

    # Limit patches to just the active source and its neighbors.
    idxs = vcat([s], neighbor_map[ts])
    patches = patches[idxs, :]

    # create smaller images for the MCMC sampler to use
    patch_images = [MCMC.patch_to_image(patches[1, i], images[i])
                    for i in 1:length(images)]

    # render a background image on the active source (first in list)
    background_images = [MCMC.render_patch_nmgy(patch_images[i], patches[1, i], neighbors)
                         for i in 1:length(patch_images)]

    # run mcmc sampler on this image/patch/background initialized at entry
    if use_ais
        mcmc_results = MCMC.run_ais(entry, patch_images, patches, background_images;
            num_temperatures=config.num_ais_temperatures,
            num_samples=config.num_ais_samples)
    else
        mcmc_results = MCMC.run_mcmc(entry, patch_images, patches, background_images)
    end

    # summary
    return mcmc_results
end


"""
Use multiple threads to process each target source with the specified
callback and write the results to a file.
"""
function one_node_single_infer(catalog::Vector{CatalogEntry},
                               patches::Matrix{ImagePatch},
                               target_sources::Vector{Int},
                               neighbor_map::Vector{Vector{Int}},
                               images::Vector{<:Image};
                               config=Config(),
                               do_vi=true)
    curr_source = 1
    last_source = length(target_sources)
    sources_lock = SpinLock()
    results_lock = SpinLock()
    if do_vi
        results = OptimizedSource[]
    else
        results = []
    end

    # iterate over sources
    function process_sources()
        tid = threadid()

        while true
            lock(sources_lock)
            ts = curr_source
            curr_source += 1
            unlock(sources_lock)
            if ts > last_source
                break
            end

            try
                if do_vi
                    result = process_source(config, ts, catalog, patches,
                                            target_sources, neighbor_map,
                                            images)
                else
                    result = process_source_mcmc(config, ts, catalog,
                                                 patches, target_sources,
                                                 neighbor_map, images)
                end

                lock(results_lock)
                push!(results, result)
                unlock(results_lock)
            catch ex
                if is_production_run || nthreads() > 1
                    Log.exception(ex)
                else
                    rethrow(ex)
                end
            end
        end
    end

    if nthreads() == 1
        process_sources()
    else
        ccall(:jl_threading_run, Void, (Any,), Core.svec(process_sources))
        ccall(:jl_threading_profile, Void, ())
    end

    return results
end

"""
Query the SDSS database for all fields that overlap the given RA, Dec range.
"""
function get_overlapping_field_extents(query::BoundingBox, strategy::SDSSIO.IOStrategy)
    f = SDSSIO.readFITS(strategy, SDSSIO.FieldExtents())

    hdu = f[2]::FITSIO.TableHDU

    # read in the entire table.
    all_run = read(hdu, "run")::Vector{Int16}
    all_camcol = read(hdu, "camcol")::Vector{UInt8}
    all_field = read(hdu, "field")::Vector{Int16}
    all_ramin = read(hdu, "ramin")::Vector{Float64}
    all_ramax = read(hdu, "ramax")::Vector{Float64}
    all_decmin = read(hdu, "decmin")::Vector{Float64}
    all_decmax = read(hdu, "decmax")::Vector{Float64}

    close(f)

    ret = Tuple{RunCamcolField, BoundingBox}[]

    # The ramin, ramax, etc is a bit unintuitive because we're looking
    # for any overlap.
    for i in eachindex(all_ramin)
        if (all_ramax[i] > query.ramin && all_ramin[i] < query.ramax &&
                all_decmax[i] > query.decmin && all_decmin[i] < query.decmax)
            cur_box = BoundingBox(all_ramin[i], all_ramax[i],
                                  all_decmin[i], all_decmax[i])
            cur_fe = RunCamcolField(all_run[i], all_camcol[i], all_field[i])
            push!(ret, (cur_fe, cur_box))
        end
    end

    return ret
end
get_overlapping_field_extents(query::BoundingBox, stagedir::String) =
    get_overlapping_field_extents(query, SDSSIO.PlainFITSStrategy(stagedir))


"""
Like `get_overlapping_field_extents()`, but return a Vector of
(run, camcol, field) triplets.
"""
function get_overlapping_fields(query::BoundingBox, stagedir)
    fes = get_overlapping_field_extents(query, stagedir)
    [fe[1] for fe in fes]
end


"""
Save provided results to a JLD file.
"""
function save_results(outdir, ramin, ramax, decmin, decmax, results)
    fname = @sprintf("%s/celeste-%.4f-%.4f-%.4f-%.4f.jld",
                     outdir, ramin, ramax, decmin, decmax)
    JLD.save(fname, "results", results)
end

save_results(outdir, box, results) =
    save_results(outdir, box.ramin, box.ramax, box.decmin, box.decmax, results)


function infer_box(images::Vector{<:Image}, box::BoundingBox; method=:joint,
                   do_vi=true, config=Config())
    # detect sources on all the images
    catalog, patches = detect_sources(images)

    # Get indices of entries in the RA/Dec range of interest.
    # (Some images can have regions that are outside the box, so not
    # all sources are necessarily in the box.)
    entry_in_range = entry->((box.ramin < entry.pos[1] < box.ramax) &&
                             (box.decmin < entry.pos[2] < box.decmax))
    target_ids = find(entry_in_range, catalog)

    # find objects with patches overlapping
    neighbor_map = [Model.find_neighbors(patches, id)
                    for id in target_ids]

    if method == :joint
        results = one_node_joint_infer(catalog, patches, target_ids,
                                       neighbor_map, images; config=config)
    elseif method == :single
        results = one_node_single_infer(catalog, patches, target_ids,
                                        neighbor_map, images; do_vi=do_vi,
                                        config=config)
    else
        error("unknown method: $method")
    end
    return results
end


# Called from main entry point.
function infer_box(strategy, box::BoundingBox, outdir::String)
    Log.info("processing box $(box.ramin), $(box.ramax), $(box.decmin), ",
             "$(box.decmax) with $(nthreads()) threads")
    @time begin
        # Get vector of (run, camcol, field) triplets overlapping this patch
        # and load images for them.
        rcfs = get_overlapping_fields(box, strategy)
        images = SDSSIO.load_field_images(strategy, rcfs)

        results = infer_box(images, box)

        save_results(outdir, box, results)
    end
end


function infer_box(box::BoundingBox, stagedir::String, outdir::String)
    strategy = PlainFITSStrategy(stagedir)
    infer_box(strategy, box, outdir)
end

end
