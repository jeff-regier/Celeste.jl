module ParallelRun

using Base.Threads
using Base.Dates

import FITSIO
import JLD
import WCS

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
# initialization helpers

"""
Computes the nearby light sources in the catalog for each of the target
sources.

Arguments:
    target_sources: indexes of astronomical objects in the catalog to infer
    catalog: astronomical objects appearing the images
    images: astronomical images
"""
function find_neighbors(target_sources::Vector{Int64},
                        catalog::Vector{CatalogEntry},
                        images::Vector{<:Image})
    psf_width_ub = zeros(NUM_BANDS)
    for img in images
        psf_width = Model.get_psf_width(img.psf)
        psf_width_ub[img.b] = max(psf_width_ub[img.b], psf_width)
    end

    epsilon_lb = fill(Inf, NUM_BANDS)
    for img in images
        epsilon = img.sky[div(img.H, 2), div(img.W, 2)]
        epsilon_lb[img.b] = min(epsilon_lb[img.b], epsilon)
    end

    radii_map = zeros(length(catalog))
    for s in 1:length(catalog)
        ce = catalog[s]
        for img in images
            radius_pix = Model.choose_patch_radius(ce, img, width_scale=1.2)
            radii_map[s] = max(radii_map[s], radius_pix)
        end
        @assert radii_map[s] <= 25
    end

    dist(ra1, dec1, ra2, dec2) = (3600 / 0.396) * max(abs(dec2 - dec1), abs(ra2 - ra1))

    neighbor_map = Vector{Int64}[Int64[] for s in target_sources]

    # If this loop isn't super fast in pratice, we can tile (the sky, not the
    # images) or build a spatial index with a library before distributing
    for ts in 1:length(target_sources)
        s = target_sources[ts]
        ce = catalog[s]

        for s2 in 1:length(catalog)
            ce2 = catalog[s2]
            ctrs_dist = dist(ce.pos[1], ce.pos[2], ce2.pos[1], ce2.pos[2])

            if s2 != s && ctrs_dist < radii_map[s] + radii_map[s2]
                push!(neighbor_map[ts], s2)
            end
        end
    end

    neighbor_map
end


"""
Record which pixels in each patch will be considered when computing the
objective function.

Non-standard arguments:
  noise_fraction: The proportion of the noise below which we will remove pixels.
"""
function load_active_pixels!(config::Config,
                             images::Vector{<:Image},
                             patches::Matrix{SkyPatch};
                             exclude_nan=true,
                             noise_fraction=0.5)
    S, N = size(patches)

    for n = 1:N, s=1:S
        img = images[n]
        p = patches[s,n]

        # (h2, w2) index the local patch, while (h, w) index the image
        H2, W2 = size(p.active_pixel_bitmap)
        for w2 in 1:W2, h2 in 1:H2
            h = p.bitmap_offset[1] + h2
            w = p.bitmap_offset[2] + w2

            # skip masked pixels
            if isnan(img.pixels[h, w]) && exclude_nan
                p.active_pixel_bitmap[h2, w2] = false
                continue
            end

            # include pixels that are close, even if they aren't bright
            sq_dist = (h - p.pixel_center[1])^2 + (w - p.pixel_center[2])^2
            if sq_dist < config.min_radius_pix^2
                p.active_pixel_bitmap[h2, w2] = true
                continue
            end

            # if this pixel is bright, let's include it
            # (in the future we may want to do something fancier, like
            # fitting an elipse, so we don't include nearby sources' pixels,
            # or adjusting active pixels during the optimization)
            # Note: This is risky because bright pixels are disproportionately likely
            # to get included, even if it's because of noise. Therefore it's important
            # to keep the noise fraction pretty low.
            threshold = img.nelec_per_nmgy[h] * img.sky[h, w] * (1. + noise_fraction)
            p.active_pixel_bitmap[h2, w2] = img.pixels[h, w] > threshold
        end
    end
end


# The range of image pixels in a vector of patches
function get_active_pixel_range(
    patches::Matrix{SkyPatch}, sources::Vector{Int}, n::Int)
    H_min = minimum([ p.bitmap_offset[1] + 1 for p in patches[sources, n] ])
    W_min = minimum([ p.bitmap_offset[2] + 1 for p in patches[sources, n] ])
    H_max = maximum([ p.bitmap_offset[1] + size(p.active_pixel_bitmap, 1)
                      for p in patches[sources, n] ])
    W_max = maximum([ p.bitmap_offset[2] + size(p.active_pixel_bitmap, 2)
                      for p in patches[sources, n] ])
    H_min, W_min, H_max, W_max
end


# Is a pixel (h, w) in whole-image coordinates an active pixel in the patch p?
function is_pixel_in_patch(h::Int, w::Int, p::SkyPatch)
    hp = h - p.bitmap_offset[1]
    wp = w - p.bitmap_offset[2]
    in_patch =
        (hp > 0) &
        (wp > 0) &
        (hp <= size(p.active_pixel_bitmap, 1)) &
        (wp <= size(p.active_pixel_bitmap, 2))
    if !in_patch
        return false
    else
        return p.active_pixel_bitmap[hp, wp]
    end
end


# return the world coordinates of all objects in the catalog as a 2xN matrix
# where `result[:, i]` is the (latitude, longitude) of the i-th object.
function _worldcoords(catalog::SEP.Catalog, wcs::WCS.WCSTransform)
    pixcoords = Array{Float64}(2, length(catalog.x))
    for i in eachindex(catalog.x)
        pixcoords[1, i] = catalog.x[i]
        pixcoords[2, i] = catalog.y[i]
    end
    return WCS.pix_to_world(wcs, pixcoords)
end


# Get angle offset between +x axis and +Dec axis (North) from a WCS transform.
# This assumes there is no skew, meaning the x and y axes are perpindicular in
# world coordinates. It is also only based on the CD matrix.
function _x_vs_n_angle(wcs::WCS.WCSTransform)
    cd = wcs[:cd]::Matrix{Float64}
    sgn = sign(det(cd))
    n_vs_y_rot = atan2(sgn * cd[1, 2],  sgn * cd[1, 1])  # angle of N CCW
                                                         # from +y axis
    return -(n_vs_y_rot + pi/2)  # angle of +x CCW from N
end


"""
    detect_sources(images)

Detect sources in a set of (possibly overlapping) `Image`s and combine
duplicates.

# Returns
"""
function detect_sources(images::Vector{<:Image})

    catalogs = SEP.Catalog[]
    for image in images

        calpixels = Model.calibrated_pixels(image)

        # Run background, just to get background rms.
        #
        # We're using sky subtracted (and calibrated) image data, but,
        # we still run a background analysis, just to determine the
        # rough image noise for the purposes of setting a
        # threshold. This could be slightly suboptimal in terms of
        # selecting faint sources: If the image noise varies
        # significantly across the image, the threshold might be too
        # high or too low in some places. However, we don't really
        # trust the variable background RMS without first masking
        # sources. (We could add this.)
        bkg = SEP.Background(calpixels; boxsize=(256, 256),
                             filtersize=(3, 3))
        noise = SEP.global_rms(bkg)
        push!(catalogs, SEP.extract(calpixels, 1.3; noise=noise))
    end

    # The image catalogs only have pixel positions. To match objects in
    # different images, we need to convert these pixel positions to world
    # coordinates in each image.
    catalogs_worldcoords = [_worldcoords(catalog, image.wcs)
                            for (catalog, image) in zip(catalogs, images)]

    # Initialize the "joined" catalog to all the coordinates in the first
    # image. `detections` tracks image index and object index of detections.
    joined_ra = length(images) > 0 ? catalogs_worldcoords[1][1, :] : Float64[]
    joined_dec = length(images) > 0 ? catalogs_worldcoords[1][2, :] : Float64[]
    detections = (length(images) > 0 ?
                  [[(1, j)] for j in 1:length(catalogs[1].x)] :
                  Vector{Tuple{Int, Int}}[])

    # Search for matches in remaining images
    for i in 2:length(images)
        ra = catalogs_worldcoords[i][1, :]
        dec = catalogs_worldcoords[i][2, :]
        idx, dist = match_coordinates(ra, dec, joined_ra, joined_dec)
        for j in eachindex(idx)
            # If there is an object in the joined catalog within 1 arcsec,
            # add the (image, object) index to detections for that object.
            # Otherwise, add the position and (image, object) index as a new
            # object in the joined catalog.
            if dist[j] < (1.0 / 3600.0)
                push!(detections[idx[j]], (i, j))
            else
                push!(joined_ra, ra[j])
                push!(joined_dec, dec[j])
                push!(detections, [(i, j)])
            end
        end
    end

    # Initialize joined output catalog and patches
    nobjects = length(joined_ra)
    result = Array{CatalogEntry}(nobjects)
    patches = Array{SkyPatch}(nobjects, length(images))

    # Precalculate some angles
    x_vs_n_angles = [_x_vs_n_angle(image.wcs) for image in images]

    # Loop over output catalog:
    # - Create catalog entry based on "best" detection.
    # - Create patches based on detection in each image.
    for i in 1:nobjects
        world_center = [joined_ra[i], joined_dec[i]]

        # which detection in each band is the "best" (has most pixels)
        # We use this to initialize flux in each band
        best = fill((0, 0), NUM_BANDS)
        npix = fill(0, NUM_BANDS)
        for (j, catidx) in detections[i]
            b = images[j].b
            np = catalogs[j].npix[catidx]
            if np > npix[b]
                best[b] = (j, catidx)
                npix[b] = np
            end
        end

        # set gal_fluxes (and star_fluxes) to best detection flux or 0 if not
        # detected.
        gal_fluxes = [j != 0 ? catalogs[j].flux[catidx] : 0.0
                      for (j, catidx) in best]
        star_fluxes = copy(gal_fluxes)

        # use the single best band for shape parameters
        j, catidx = best[indmax(npix)]
        gal_axis_ratio = catalogs[j].b[catidx] / catalogs[j].a[catidx]

        # SEP angle is CCW from +x axis and in [-pi/2, pi/2].
        # Add offset of x axis from N to make angle CCW from N
        gal_angle = catalogs[j].theta[catidx] + x_vs_n_angles[j]

        # A 2-d symmetric gaussian has CDF(r) =  1 - exp(-(r/sigma)^2/2)
        # The half-light radius is then r = sigma * sqrt(2 ln(2))
        sigma = sqrt(catalogs[j].a[catidx] * catalogs[j].b[catidx])
        gal_radius_px = sigma * sqrt(2.0 * log(2.0))

        result[i] = CatalogEntry(world_center,
                                 false,  # is_star
                                 star_fluxes,
                                 gal_fluxes,
                                 0.5,  # gal_frac_dev
                                 gal_axis_ratio,
                                 gal_angle,
                                 gal_radius_px)

        # create patches based on detections
        for (j, catidx) in detections[i]
            box = (Int(catalogs[j].xmin[catidx]):Int(catalogs[j].xmax[catidx]),
                   Int(catalogs[j].ymin[catidx]):Int(catalogs[j].ymax[catidx]))
            box = Model.dilate_box(box, 0.2)
            minbox = Model.box_around_point(images[j].wcs, world_center, 5.0)
            patches[i, j] = SkyPatch(images[j], Model.enclose_boxes(box, minbox))
        end

        # fill patches for images where there was no detection
        for j in 1:length(images)
            if !isassigned(patches, i, j)
                box = Model.box_around_point(images[j].wcs, world_center, 5.0)
                patches[i, j] = SkyPatch(images[j], box)
            end
        end
    end

    return result, patches
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
                        patches::Matrix{SkyPatch},
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
                             patches::Matrix{SkyPatch},
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
                               patches::Matrix{SkyPatch},
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
