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
import ..Infer
import ..SDSSIO: RunCamcolField, IOStrategy, PlainFITSStrategy
import ..PSF
import ..SEP
import ..Coordinates: angular_separation, match_coordinates

import ..DeterministicVI: infer_source

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
    detect_sources(images)

Detect sources in a set of (possibly overlapping) `RawImage`s and combine
duplicates.

# Returns

- `catalog::Vector{CatalogEntry}`: Detected sources.
- `source_radii::Vector{Float64}`: Radius of circle containing all of each
  source's member pixels (degrees).

# Notes

Here, we're dealing with the `RawImage` type, which is SDSS-specific. The
pixel values are already sky subtracted (and calibrated), so we don't have to
background subtract. However, we still run a background analysis, just to
determine the rough image noise for the purposes of thresholding.

This could be slightly suboptimal in terms of
selecting faint sources: If the image noise varies significantly
across the image, the threshold might be too high or too low in some
places. However, we don't really trust the variable background RMS without
first masking sources. (We could add this.)
"""
function detect_sources(images::Vector{SDSSIO.RawImage})

    catalog = Vector{CatalogEntry}()
    source_radii = Vector{Float64}()

    for image in images

        # Run background, just to get background rms.
        bkg = SEP.Background(image.pixels; boxsize=(256, 256),
                             filtersize=(3, 3))
        sep_catalog = SEP.extract(image.pixels, 1.3; noise=SEP.global_rms(bkg))

        # convert pixel coordinates to world coordinates
        pixcoords = Array{Float64}(2, length(sep_catalog.x))
        for i in eachindex(sep_catalog.x)
            pixcoords[1, i] = sep_catalog.x[i]
            pixcoords[2, i] = sep_catalog.y[i]
        end
        worldcoords = WCS.pix_to_world(image.wcs, pixcoords)

        # Get angle offset between +RA axis and +x axis from the
        # image's WCS.  This assumes there is no skew, meaning the x
        # and y axes are perpindicular in world coordinates.
        cd = image.wcs[:cd]
        sgn = sign(det(cd))
        n_vs_y_rot = atan2(sgn * cd[1, 2],  sgn * cd[1, 1])  # angle of N CCW
                                                             # from +y axis
        x_vs_n_rot = -(n_vs_y_rot + pi/2)  # angle of +x CCW from N

        # convert sep_catalog entries to CatalogEntries
        im_catalog = Vector{CatalogEntry}(length(sep_catalog.npix))
        im_source_radii = Vector{Float64}(length(sep_catalog.npix))
        for i in eachindex(sep_catalog.npix)

            # For galaxy flux, use sum of all pixels above
            # threshold. Set other band fluxes to NaN, to indicate which
            # band the flux is measured in. We'll use this information below
            # when merging detections in different bands.
            gal_fluxes = fill(NaN, NUM_BANDS)
            gal_fluxes[image.b] = sep_catalog.flux[i]

            gal_ab = sep_catalog.a[i] / sep_catalog.b[i]

            # SEP angle is CCW from +x axis and in [-pi/2, pi/2].
            # Add offset of x axis from N to make angle CCW from N
            gal_angle = sep_catalog.theta[i] + x_vs_n_rot

            # A 2-d symmetric gaussian has CDF(r) =  1 - exp(-(r/sigma)^2/2)
            # The half-light radius is then r = sigma * sqrt(2 ln(2))
            sigma = sqrt(sep_catalog.a[i] * sep_catalog.b[i])
            gal_scale = sigma * sqrt(2. * log(2.))

            pos = worldcoords[:, i]
            im_catalog[i] = CatalogEntry(pos,
                                         false,  # is_star
                                         Float64[],  # will replace below
                                         gal_fluxes,
                                         0.5,  # gal_frac_dev
                                         gal_ab,
                                         gal_angle,
                                         gal_scale)

            # get object extent in degrees from bounding box in pixels
            xmin = sep_catalog.xmin[i]
            xmax = sep_catalog.xmax[i]
            ymin = sep_catalog.ymin[i]
            ymax = sep_catalog.ymax[i]
            corner_pixcoords = Float64[xmin xmin xmax xmax;
                                       ymin ymax ymin ymax]
            corners = WCS.pix_to_world(image.wcs, corner_pixcoords)
            im_source_radii[i] =
                maximum(angular_separation(pos[1], pos[2],
                                           corners[1, j], corners[2, j])
                        for j in 1:4)
        end

        # Combine the catalog for this image with the joined catalog
        if length(catalog) == 0
            catalog = im_catalog
            source_radii = im_source_radii
        else
            # for each detection in image, find nearest match in joined catalog
            idx, dist = match_coordinates([ce.pos[1] for ce in im_catalog],
                                          [ce.pos[2] for ce in im_catalog],
                                          [ce.pos[1] for ce in catalog],
                                          [ce.pos[2] for ce in catalog])

            for (i, ce) in enumerate(im_catalog)
                # if there is an existing object within 1 arcsec,
                # add it to the joined catalog
                if dist[i] < (1.0 / 3600.)
                    existing_ce = catalog[idx[i]]
                    if isnan(existing_ce.gal_fluxes[image.b])
                        existing_ce.gal_fluxes[image.b] =
                            ce.gal_fluxes[image.b]
                    end
                    if im_source_radii[i] > source_radii[idx[i]]
                        source_radii[idx[i]] = im_source_radii[i]
                    end

                # if not, add entry to joined catalog
                else
                    push!(catalog, ce)
                    push!(source_radii, im_source_radii[i])
                end
            end
        end
    end  # loop over images

    # clean up NaN flux entries in joined catalog,
    # copy gal_fluxes to star_fluxes
    for ce in catalog
        minflux = minimum(f for f in ce.gal_fluxes if !isnan(f))
        for i in eachindex(ce.gal_fluxes)
            if isnan(ce.gal_fluxes[i])
                ce.gal_fluxes[i] = minflux
            end
        end
        ce.star_fluxes = copy(ce.gal_fluxes)
    end

    return catalog, source_radii
end


"""
New version of infer_init() that uses detect_sources() instead of SDSS
photoObj catalog for initialization.
"""
function infer_init_new(rcfs::Vector{RunCamcolField},
                        strategy::SDSSIO.IOStrategy;
                        box=BoundingBox(-1000., 1000., -1000., 1000.))

    # Initialize variables to empty vectors in case try block fails
    catalog = CatalogEntry[]
    source_radii = Float64[]
    target_sources = Int[]
    images = Image[]

    try
        # Read in images for all RCFs
        raw_images = SDSSIO.load_raw_images(strategy, rcfs)

        # detect sources on all raw images (before background added back)
        catalog, source_radii = detect_sources(raw_images)

        # Get indices of entries in the RA/Dec range of interest.
        # (Some images can have regions that are outside the box, so not
        # all sources are necessarily in the box.)
        entry_in_range = entry->((box.ramin < entry.pos[1] < box.ramax) &&
                                 (box.decmin < entry.pos[2] < box.decmax))
        target_sources = find(entry_in_range, catalog)

        Log.info("$(Time(now())): $(length(catalog)) primary sources, ",
                 "$(length(target_sources)) target sources in $(box.ramin), ",
                 "$(box.ramax), $(box.decmin), $(box.decmax)")

        # convert raw images to images
        try
            images = [convert(Image, raw_image) for raw_image in raw_images]
        catch exc
            Log.exception(exc)
            rethrow()
        end

    catch ex
        Log.exception(ex)
        empty!(target_sources)
        rethrow()
    end

    # build neighbor map based on source radii
    neighbor_map = Vector{Int64}[Int64[] for s in target_sources]
    for ts in 1:length(target_sources)
        s = target_sources[ts]
        ce = catalog[s]
        ce_rad = source_radii[s]

        for s2 in 1:length(catalog)
            s2 == s && continue
            ce2 = catalog[s2]
            dist = angular_separation(ce.pos[1], ce.pos[2],
                                      ce2.pos[1], ce2.pos[2])

            if dist < source_radii[s] + source_radii[s2]
                push!(neighbor_map[ts], s2)
            end
        end
    end

    return catalog, target_sources, neighbor_map, images
end


"""
Given a list of RCFs, load the catalogs, determine the target sources,
load the images, and build the neighbor map.
"""
function infer_init(rcfs::Vector{RunCamcolField},
                    strategy::SDSSIO.IOStrategy;
                    box=BoundingBox(-1000., 1000., -1000., 1000.),
                    primary_initialization=true)
    catalog = Vector{CatalogEntry}()
    target_sources = Vector{Int}()
    neighbor_map = Vector{Vector{Int}}()
    images = Vector{Image}()

    # Read all primary objects in these fields.
    duplicate_policy = primary_initialization ? :primary : :first

    for rcf in rcfs
        this_cat = SDSSIO.read_photoobj_files(strategy, [rcf],
                                              duplicate_policy=duplicate_policy)
        append!(catalog, this_cat)
    end

    # Get indices of entries in the RA/Dec range of interest.
    entry_in_range = entry->((box.ramin < entry.pos[1] < box.ramax) &&
                             (box.decmin < entry.pos[2] < box.decmax))
    target_sources = find(entry_in_range, catalog)

    Log.info("$(Time(now())): $(length(catalog)) primary sources, ",
             "$(length(target_sources)) target sources in $(box.ramin), ",
             "$(box.ramax), $(box.decmin), $(box.decmax)")

    # Load images and neighbor map for target sources
    if length(target_sources) > 0
        # Read in images for all (run, camcol, field).
        try
            images = SDSSIO.load_field_images(strategy, rcfs)
        catch ex
            Log.exception(ex)
            empty!(target_sources)
        end

        neighbor_map = Infer.find_neighbors(target_sources, catalog, images)
    end

    return catalog, target_sources, neighbor_map, images
end


# ------
# optimization result container

struct OptimizedSource
    init_ra::Float64
    init_dec::Float64
    vs::Vector{Float64}
end

function serialize(s::Base.AbstractSerializer, os::OptimizedSource)
    Base.serialize_type(s, typeof(os))
    write(s.io, os.init_ra)
    write(s.io, os.init_dec)
    for i = 1:length(Celeste.Model.ids)
        write(s.io, os.vs[i])
    end
end

function deserialize(s::Base.AbstractSerializer, ::Type{OptimizedSource})
    init_ra = read(s.io, Float64)::Float64
    init_dec = read(s.io, Float64)::Float64
    vs = zeros(Float64, length(Celeste.Model.ids))
    for i = 1:length(Celeste.Model.ids)
        vs[i] = read(s.io, Float64)
    end
    OptimizedSource(init_ra, init_dec, vs)
end


# ------
# optimization

"""
Optimize the `ts`th element of `target_sources`.
"""
function process_source(config::Config,
                        ts::Int,
                        catalog::Vector{CatalogEntry},
                        target_sources::Vector{Int},
                        neighbor_map::Vector{Vector{Int}},
                        images::Vector{Image})
    s = target_sources[ts]
    entry = catalog[s]
    neighbors = catalog[neighbor_map[ts]]

    tic()
    vs_opt = infer_source(config, images, neighbors, entry)
    Log.info("#$(ts) at ($(entry.pos[1]), $(entry.pos[2])): $(toq()) secs")
    return OptimizedSource(entry.pos[1], entry.pos[2], vs_opt)
end


"""
Use multiple threads to process each target source with the specified
callback and write the results to a file.
"""
function one_node_single_infer(catalog::Vector{CatalogEntry},
                               target_sources::Vector{Int},
                               neighbor_map::Vector{Vector{Int}},
                               images::Vector{Image};
                               infer_source_callback=infer_source,
                               config=Config())
    curr_source = 1
    last_source = length(target_sources)
    sources_lock = SpinLock()
    results = OptimizedSource[]
    results_lock = SpinLock()

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
                result = process_source(config, ts, catalog, target_sources,
                                        neighbor_map, images)

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
Store the contents of the `field_extents.fits` file, so it doesn't
have to be loaded repeatedly.
"""
struct FieldExtent
    run::Int16
    camcol::UInt8
    field::Int16
    ramin::Float64
    ramax::Float64
    decmin::Float64
    decmax::Float64
end


"""
Load `field_extents.fits` from `stagedir` and return a vector of
`FieldExtent`s.
"""
function load_field_extents(strategy)
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

    fes = Vector{FieldExtent}()
    for i = 1:length(all_run)
        fe = FieldExtent(all_run[i], all_camcol[i], all_field[i],
                         all_ramin[i], all_ramax[i],
                         all_decmin[i], all_decmax[i])
        push!(fes, fe)
    end

    return fes
end


"""
Use the provided `FieldExtent`s to return a list of RCFs that overlap
the specified box.
"""
function get_overlapping_fields(query::BoundingBox, fes::Vector{FieldExtent})
    # The ramin, ramax, etc is a bit unintuitive because we're looking
    # for any overlap.
    rcfs = Vector{RunCamcolField}()
    for i in eachindex(fes)
        if (fes[i].ramax > query.ramin &&
                fes[i].ramin < query.ramax &&
                fes[i].decmax > query.decmin &&
                fes[i].decmin < query.decmax)
            push!(rcfs, RunCamcolField(fes[i].run, fes[i].camcol, fes[i].field))
        end
    end

    return rcfs
end


"""
Use the provided `FieldExtent`s to return a list of RCFs that overlap
all the provided boxes.
"""
function get_overlapping_fields(queries::Vector{Vector{BoundingBox}},
                                fes::Vector{FieldExtent})
    rcfs = Vector{RunCamcolField}()

    for boxlist in queries
        for query in boxlist
            # The ramin, ramax, etc is a bit unintuitive because we're looking
            # for any overlap.
            for i in eachindex(fes)
                if (fes[i].ramax > query.ramin &&
                        fes[i].ramin < query.ramax &&
                        fes[i].decmax > query.decmin &&
                        fes[i].decmin < query.decmax)
                    push!(rcfs, RunCamcolField(fes[i].run,
                                               fes[i].camcol,
                                               fes[i].field))
                end
            end
        end
        rcfs = unique(rcfs)
    end

    return rcfs
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


"""
called from main entry point.
"""
function infer_box(strategy, box::BoundingBox, outdir::String)
    Log.info("processing box $(box.ramin), $(box.ramax), $(box.decmin), ",
             "$(box.decmax) with $(nthreads()) threads")
    @time begin
        # Get vector of (run, camcol, field) triplets overlapping this patch
        rcfs = get_overlapping_fields(box, strategy)
        catalog, target_sources, neighbor_map, images =
            infer_init(rcfs, strategy; box=box, primary_initialization=true)
        results = one_node_joint_infer(catalog, target_sources, neighbor_map, images)
        save_results(outdir, box, results)
    end
end

function infer_box(box::BoundingBox, stagedir::String, outdir::String)
    strategy = PlainFITSStrategy(stagedir)
    infer_box(strategy, box, outdir)
end

end
