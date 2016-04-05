# Functions for interacting with Celeste from the command line.

import FITSIO
import JLD
import Logging  # just for testing right now

using .Types
import .SDSSIO
import .ModelInit
import .OptimizeElbo

const TILE_WIDTH = 20
const DEFAULT_MAX_ITERS = 50
const MIN_FLUX = 2.0

# Use distributed parallelism (with Dtree)
if haskey(ENV, "USE_DTREE") && ENV["USE_DTREE"] != ""
    const Distributed = true
    using Dtree
else
    const Distributed = false
    const dt_nodeid = 1
    const dt_nnodes = 1
    DtreeScheduler(n, f) = ()
    initwork(dt) = 0, (1, 0)
    getwork(dt) = 0, (1, 0)
    runtree(dt) = 0
    cpu_pause() = ()
end

# Use threads (on the loop over sources)
const Threaded = false
if Threaded && VERSION > v"0.5.0-dev"
    using Base.Threads
else
    # Pre-Julia 0.5 there are no threads
    nthreads() = 1
    threadid() = 1
    macro threads(x)
        x
    end
    SpinLock() = 1
    lock!(l) = ()
    unlock!(l) = ()
end

# A workitem is of this ra / dec size
const wira = 0.03
const widec = 0.03

"""
Timing information.
"""
type InferTiming
    num_infers::Int64
    read_photoobj::Float64
    read_img::Float64
    init_mp::Float64
    fit_psf::Float64
    opt_srcs::Float64
    num_srcs::Int64
    write_results::Float64
    wait_done::Float64

    InferTiming() = new(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)
end

function add_timing!(i::InferTiming, j::InferTiming)
    i.num_infers = i.num_infers + j.num_infers
    i.read_photoobj = i.read_photoobj + j.read_photoobj
    i.read_img = i.read_img + j.read_img
    i.init_mp = i.init_mp + j.init_mp
    i.fit_psf = i.fit_psf + j.fit_psf
    i.opt_srcs = i.opt_srcs + j.opt_srcs
    i.num_srcs = i.num_srcs + j.num_srcs
    i.write_results = i.write_results + j.write_results
    i.wait_done = i.wait_done + j.wait_done
end

"""
An area of the sky subtended by `ramin`, `ramax`, `decmin`, and `decmax`.
"""
type SkyArea
    ramin::Float64
    ramax::Float64
    decmin::Float64
    decmax::Float64
    nra::Int64
    ndec::Int64
end

"""
Given a SkyArea that is to be divided into `skya.nra` x `skya.ndec` patches,
return the `i`th patch. `i` is a linear index between 1 and
`skya.nra * skya.ndec`.

This function assumes a cartesian (rather than spherical) coordinate system!
"""
function divide_skyarea(skya::SkyArea, i)
    global wira, widec
    ix, iy = ind2sub((skya.nra, skya.ndec), i)

    return (skya.ramin + (ix - 1) * wira,
            min(skya.ramin + ix * wira, skya.ramax),
            skya.decmin + (iy - 1) * widec,
            min(skya.decmin + iy * widec, skya.decmax))
end

@inline nputs(nid, s) = ccall(:puts, Cint, (Ptr{Int8},), string("[$nid] ", s))
@inline phalse(b) = b[] = false

function time_puts(elapsedtime, bytes, gctime, allocs)
    s = @sprintf("%10.6f seconds", elapsedtime/1e9)
    if bytes != 0 || allocs != 0
        bytes, mb = Base.prettyprint_getunits(bytes, length(Base._mem_units),
                            Int64(1024))
        allocs, ma = Base.prettyprint_getunits(allocs, length(Base._cnt_units),
                            Int64(1000))
        if ma == 1
            s = string(s, @sprintf(" (%d%s allocation%s: ", allocs,
                                   Base._cnt_units[ma], allocs==1 ? "" : "s"))
        else
            s = string(s, @sprintf(" (%.2f%s allocations: ", allocs,
                                   Base._cnt_units[ma]))
        end
        if mb == 1
            s = string(s, @sprintf("%d %s%s", bytes,
                                   Base._mem_units[mb], bytes==1 ? "" : "s"))
        else
            s = string(s, @sprintf("%.3f %s", bytes, Base._mem_units[mb]))
        end
        if gctime > 0
            s = string(s, @sprintf(", %.2f%% gc time", 100*gctime/elapsedtime))
        end
        s = string(s, ")")
    elseif gctime > 0
        s = string(s, @sprintf(", %.2f%% gc time", 100*gctime/elapsedtime))
    end
    nputs(dt_nodeid, s)
end

function set_logging_level(level)
    if level == "OFF"
      Logging.configure(level=Logging.OFF)
    elseif level == "DEBUG"
      Logging.configure(level=Logging.DEBUG)
    elseif level == "INFO"
      Logging.configure(level=Logging.INFO)
    elseif level == "WARNING"
      Logging.configure(level=Logging.WARNING)
    elseif level == "ERROR"
      Logging.configure(level=Logging.ERROR)
    elseif level == "CRITICAL"
      Logging.configure(level=Logging.CRITICAL)
    else
      Logging.err("Unknown logging level $(level)")
    end
end


"""
read_photoobj_files(fieldids, dirs) -> Vector{CatalogEntry}

Combine photoobj catalogs for the given overlapping fields, returning a single
joined catalog.

The `duplicate_policy` argument controls how catalogs are joined.
With `duplicate_policy = :primary`, only primary objects are included in the
combined catalog.
With `duplicate_policy = :first`, only the first detection is included in the
combined catalog.
"""
function read_photoobj_files(fieldids::Vector{Tuple{Int, Int, Int}}, dirs;
        duplicate_policy=:primary)
    @assert length(fieldids) == length(dirs)
    @assert duplicate_policy == :primary || duplicate_policy == :first
    @assert duplicate_policy == :primary || length(dirs) == 1

    Logging.info("reading photoobj catalogs for ", length(fieldids), " fields")

    # the code below assumes there is at least one field.
    if length(fieldids) == 0
        return CatalogEntry[]
    end

    # Read in all photoobj catalogs.
    rawcatalogs = Array(Dict, length(fieldids))
    for i in eachindex(fieldids)
        run, camcol, field = fieldids[i]
        dir = dirs[i]
        fname = @sprintf "%s/photoObj-%06d-%d-%04d.fits" dir run camcol field
        Logging.info("field $(fieldids[i]): reading $fname")
        rawcatalogs[i] = SDSSIO.read_photoobj(fname)
    end

    for i in eachindex(fieldids)
        Logging.info("field ", fieldids[i], ": ", length(rawcatalogs[i]["objid"]),
             " entries")
    end

    # Limit each catalog to primary objects and objects where thing_id != -1
    # (thing_id == -1 indicates that the matching process failed)
    for cat in rawcatalogs
        mask = (cat["thing_id"] .!= -1)
        if duplicate_policy == :primary
            mask &= (cat["mode"] .== 0x01)
        end
        for key in keys(cat)
            cat[key] = cat[key][mask]
        end
    end

    for i in eachindex(fieldids)
        Logging.info("field ", fieldids[i], ": ", length(rawcatalogs[i]["objid"]),
             " filtered entries")
    end

    # Merge all catalogs together (there should be no duplicate objects,
    # because for each object there should only be one "primary" occurance.)
    rawcatalog = deepcopy(rawcatalogs[1])
    for i=2:length(rawcatalogs)
        for key in keys(rawcatalog)
            append!(rawcatalog[key], rawcatalogs[i][key])
        end
    end

    # check that there are no duplicate thing_ids (see above comment)
    if length(Set(rawcatalog["thing_id"])) < length(rawcatalog["thing_id"])
        error("Found one or more duplicate primary thing_ids in photoobj " *
              "catalogs")
    end

    # convert to celeste format catalog
    catalog = ModelInit.convert(Vector{CatalogEntry}, rawcatalog)

    return catalog
end


"""
Divide the given ra, dec range into sky areas of `wira`x`widec` and
use Dtree to distribute these sky areas to nodes. Within each node
use `infer()` to fit the Celeste model to sources in each sky area.
"""
function divide_and_infer(fieldids::Vector{Tuple{Int, Int, Int}},
                          frame_dirs::Vector;
                          objid="",
                          fpm_dirs=frame_dirs,
                          psfield_dirs=frame_dirs,
                          photofield_dirs=frame_dirs,
                          photoobj_dirs=frame_dirs,
                          ra_range=(-1000., 1000.),
                          dec_range=(-1000., 1000.),
                          primary_initialization=true,
                          max_iters=DEFAULT_MAX_ITERS,
                          timing=InferTiming(),
                          outdir=".",
                          output_results=save_results)
    if dt_nodeid == 1
        nputs(dt_nodeid, "running on $dt_nnodes nodes")
    end

    # how many `wira` X `widec` sky areas (work items)?
    global wira, widec
    nra = ceil(Int64, (ra_range[2] - ra_range[1]) / wira)
    ndec = ceil(Int64, (dec_range[2] - dec_range[1]) / widec)
    skya = SkyArea(ra_range[1], ra_range[2], dec_range[1], dec_range[2], nra, ndec)

    num_work_items = nra * ndec
    each = ceil(Int64, num_work_items / dt_nnodes)

    if dt_nodeid == 1
        nputs(dt_nodeid, "work item dimensions: $wira X $widec")
        nputs(dt_nodeid, "$num_work_items work items, ~$each per node")
    end

    # create Dtree and get the initial allocation
    dt, is_parent = DtreeScheduler(num_work_items, 0.4)
    ni, (ci, li) = initwork(dt)
    rundt = Ref(runtree(dt))
    @inline function rundtree(again)
        if again[]
            again[] = runtree(dt)
            cpu_pause()
        end
        again[]
    end

    # work item processing loop
    nputs(dt_nodeid, "initially $ni work items ($ci to $li)")
    itimes = InferTiming()
    while ni > 0
        li == 0 && break
        if ci > li
            nputs(dt_nodeid, "consumed allocation (last was $li)")
            ni, (ci, li) = getwork(dt)
            nputs(dt_nodeid, "got $ni work items ($ci to $li)")
            continue
        end
        item = ci
        ci = ci + 1

        # map item to subarea
        iramin, iramax, idecmin, idecmax = divide_skyarea(skya, item)

        # run inference for this subarea
        results = infer(fieldids, frame_dirs;
                        ra_range=(iramin, iramax),
                        dec_range=(idecmin, idecmax),
                        fpm_dirs=frame_dirs,
                        psfield_dirs=frame_dirs,
                        photoobj_dirs=frame_dirs,
                        photofield_dirs=photofield_dirs,
                        reserve_thread=rundt,
                        thread_fun=rundtree,
                        timing=itimes)
        tic()
        output_results(outdir, iramin, iramax, idecmin, idecmax, results)
        itimes.write_results = toq()

        add_timing!(timing, itimes)
        rundtree(rundt)
    end
    nputs(dt_nodeid, "out of work")
    tic()
    while rundt[]
        rundtree(rundt)
        cpu_pause()
    end
    finalize(dt)
    timing.wait_done = toq()
end


"""
Update ModelParams with the PSFs for a range of object ids.

Args:
  - mp: A ModelParams whose patches will be updated.
  - relevant_sources: A vector of source ids that index into mp.patches
  - blob: A vector of images.

Returns:
  - Updates mp.patches in place with fitted psfs for each source in
    relevant_sources.

TODO: Unify with ModelInit.fit_object_psfs! when threading is part of all of
      Celeste.
"""
function fit_object_psfs_threaded!{NumType <: Number}(
    mp::ModelParams{NumType}, target_sources::Vector{Int}, blob::Blob)

  # Initialize a vector of optimizers and transforms.
  initial_psf_params = PSF.initialize_psf_params(psf_K, for_test=false);
  psf_transform = PSF.get_psf_transform(initial_psf_params);
  psf_optimizer_vec = Array(PSF.PsfOptimizer, nthreads())
  for tid in 1:nthreads()
    psf_optimizer_vec[tid] = PSF.PsfOptimizer(deepcopy(psf_transform), psf_K);
  end

  @assert size(mp.patches, 2) == length(blob)

  for b in 1:length(blob)  # loop over images
    Logging.debug("Fitting PSFS for band $b")
    # Get a starting point in the middle of the image.
    pixel_loc = Float64[ blob[b].H / 2.0, blob[b].W / 2.0 ]
    raw_central_psf = blob[b].raw_psf_comp(pixel_loc[1], pixel_loc[2])
    central_psf, central_psf_params =
      PSF.fit_raw_psf_for_celeste(raw_central_psf, psf_optimizer_vec[1], initial_psf_params)

    # Get all relevant sources *in this image*
    relevant_sources = ModelInit.get_all_relevant_sources_in_image(mp, target_sources, b)

    mp_lock = SpinLock()
    @threads for s in relevant_sources
      tid = threadid()
      Logging.debug("Thread $tid: fitting PSF for b=$b, source=$s, objid=$(mp.objids[s])")
      patch = mp.patches[s, b]

      # Set the starting point at the center's PSF.
      psf, psf_params =
        PSF.get_source_psf(
          patch.center, blob[b], psf_optimizer_vec[tid], central_psf_params)
      new_patch = ModelInit.SkyPatch(patch, psf);

      # Copy to the global array of patches.
      lock!(mp_lock)
      mp.patches[s, b] = new_patch
      unlock!(mp_lock)
    end
  end
end


"""
Fit the Celeste model to sources in a given ra, dec range,
based on data from specified fields

- ra_range: minimum and maximum RA of sources to consider.
- dec_range: minimum and maximum Dec of sources to consider.
- fieldids: Array of run, camcol, field triplets that the source occurs in.
- frame_dirs: Directories in which to find each field's frame FITS files.

Returns:

- Dictionary of results, keyed by SDSS thing_id.
"""
function infer(fieldids::Vector{Tuple{Int, Int, Int}},
               frame_dirs::Vector;
               objid="",
               fpm_dirs=frame_dirs,
               psfield_dirs=frame_dirs,
               photofield_dirs=frame_dirs,
               photoobj_dirs=frame_dirs,
               ra_range=(-1000., 1000.),
               dec_range=(-1000., 1000.),
               primary_initialization=true,
               max_iters=DEFAULT_MAX_ITERS,
               reserve_thread=Ref(false),
               thread_fun=phalse,
               timing=InferTiming())

    Logging.info("Running with $(nthreads()) threads")

    # Read all primary objects in these fields.
    tic()
    duplicate_policy = primary_initialization ? :primary : :first
    catalog = read_photoobj_files(fieldids, photoobj_dirs,
                        duplicate_policy=duplicate_policy)
    timing.read_photoobj = toq()
    Logging.info("$(length(catalog)) primary sources")

    reserve_thread[] && thread_fun(reserve_thread)

    # Filter out low-flux objects in the catalog.
    catalog = filter(entry->(maximum(entry.star_fluxes) >= MIN_FLUX), catalog)
    Logging.info("$(length(catalog)) primary sources after MIN_FLUX cut")

    # Filter any object not specified, if an objid is specified
    if objid != ""
        Logging.info(catalog[1].objid)
        catalog = filter(entry->(entry.objid == objid), catalog)
    end

    # Get indicies of entries in the  RA/Dec range of interest.
    entry_in_range = entry->((ra_range[1] < entry.pos[1] < ra_range[2]) &&
                             (dec_range[1] < entry.pos[2] < dec_range[2]))
    target_sources = find(entry_in_range, catalog)

    nputs(dt_nodeid, string("processing $(length(target_sources)) sources in ",
          "$(ra_range[1]), $(ra_range[2]), $(dec_range[1]), $(dec_range[2])"))

    # If there are no objects of interest, return early.
    if length(target_sources) == 0
        return Dict{Int, Dict}()
    end

    reserve_thread[] && thread_fun(reserve_thread)

    # Read in images for all (run, camcol, field).
    images = Image[]
    image_names = ASCIIString[]
    image_count = 0
    tic()
    for i in 1:length(fieldids)
        Logging.info("reading field ", fieldids[i])
        run, camcol, field = fieldids[i]
        fieldims = ModelInit.read_sdss_field(run, camcol, field, frame_dirs[i],
                                             fpm_dir=fpm_dirs[i],
                                             psfield_dir=psfield_dirs[i],
                                             photofield_dir=photofield_dirs[i])
        for b=1:length(fieldims)
          image_count += 1
          push!(image_names,
                "$image_count run=$run camcol=$camcol $field=field b=$b")
        end
        append!(images, fieldims)
    end
    timing.read_img = toq()

    reserve_thread[] && thread_fun(reserve_thread)

    Logging.debug("Image names:")
    Logging.debug(image_names)

    # initialize tiled images and model parameters for trimming.  We will
    # initialize the psf again before fitting, so we don't do it here.
    Logging.info("initializing celeste without PSF fit")
    tic()
    tiled_images = ModelInit.break_blob_into_tiles(images, TILE_WIDTH)
    mp = ModelInit.initialize_model_params(tiled_images, images, catalog,
                                           fit_psf=false)
    timing.init_mp = toq()

    reserve_thread[] && thread_fun(reserve_thread)

    # get indicies of all sources relevant to those we're actually
    # interested in, and fit a local PSF for those sources (since we skipped
    # fitting the PSF for the whole catalog above)
    Logging.info("fitting PSF for all relevant sources")
    tic()
    if Threaded
      fit_object_psfs_threaded!(mp, target_sources, images)
    else
      ModelInit.fit_object_psfs!(mp, target_sources, images)
    end
    timing.fit_psf = toq()

    reserve_thread[] && thread_fun(reserve_thread)

    # iterate over sources
    results = Dict{Int, Dict}()
    results_lock = SpinLock()
    tic()
    @threads for s in target_sources
        tid = threadid()
        if reserve_thread[] && tid == 1
            while reserve_thread[]
                thread_fun(reserve_thread)
                cpu_pause()
            end
        else
            entry = catalog[s]

            try
                nputs(dt_nodeid, "processing source $s: objid = $(entry.objid)")
                #tic()

                t0 = time()
                relevant_sources = ModelInit.get_relevant_sources(mp, s)
                mp_source = ModelParams(mp, relevant_sources)
                sa = findfirst(relevant_sources, s)
                mp_source.active_sources = Int[ sa ]
                trimmed_tiled_images =
                  ModelInit.trim_source_tiles(sa, mp_source, tiled_images;
                                              noise_fraction=0.1)
                init_time = time() - t0

                #t = toq()
                #nputs(dt_nodeid, "trimmed $s in $t secs, optimizing")
                #tic()

                t0 = time()
                iter_count, max_f, max_x, result =
                    OptimizeElbo.maximize_f(ElboDeriv.elbo, trimmed_tiled_images,
                                            mp_source;
                                            verbose=false, max_iters=max_iters)
                fit_time = time() - t0

                #t = toq()
                #nputs(dt_nodeid, "optimized $s in $t secs, writing results")

                lock!(results_lock)
                results[entry.thing_id] = Dict("objid"=>entry.objid,
                                               "ra"=>entry.pos[1],
                                               "dec"=>entry.pos[2],
                                               "vs"=>mp_source.vp[sa],
                                               "init_time"=>init_time,
                                               "fit_time"=>fit_time)
                unlock!(results_lock)
            catch ex
                Logging.err(ex)
            end
        end
    end
    timing.opt_srcs = toq()
    timing.num_srcs = length(target_sources)
    timing.num_infers = 1

    results
end


# -----------------------------------------------------------------------------
# NERSC-specific functions

"""
Query the SDSS database for all fields that overlap the given RA, Dec range.
"""
function query_overlapping_fields(ramin, ramax, decmin, decmax)

    fname = "/project/projectdirs/dasrepo/celeste-sc16/field_extents.fits"

    f = FITSIO.FITS(fname)
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

    # initialize output "table"
    out = Dict{ASCIIString, Vector}("run"=>Int[],
                                    "camcol"=>Int[],
                                    "field"=>Int[],
                                    "ramin"=>Float64[],
                                    "ramax"=>Float64[],
                                    "decmin"=>Float64[],
                                    "decmax"=>Float64[])

    # The ramin, ramax, etc is a bit unintuitive because we're looking
    # for any overlap.
    for i in eachindex(all_ramin)
        if (all_ramax[i] > ramin && all_ramin[i] < ramax &&
            all_decmax[i] > decmin && all_decmin[i] < decmax)
            push!(out["run"], all_run[i])
            push!(out["camcol"], all_camcol[i])
            push!(out["field"], all_field[i])
            push!(out["ramin"], all_ramin[i])
            push!(out["ramax"], all_ramax[i])
            push!(out["decmin"], all_decmin[i])
            push!(out["decmax"], all_decmax[i])
        end
    end

    return out
end

"""
query_overlapping_fieldids(ramin, ramax, decmin, decmax) -> Vector{Tuple{Int, Int, Int}}

Like `query_overlapping_fields`, but return a Vector of
(run, camcol, field) triplets.
"""
function query_overlapping_fieldids(ramin, ramax, decmin, decmax)
    fields = query_overlapping_fields(ramin, ramax, decmin, decmax)
    return Tuple{Int, Int, Int}[(fields["run"][i],
                                 fields["camcol"][i],
                                 fields["field"][i])
                                for i in eachindex(fields["run"])]
end


# NERSC source directories
const NERSC_DATA_ROOT = "/global/projecta/projectdirs/sdss/data/sdss/dr12/boss"
nersc_photoobj_dir(run::Integer, camcol::Integer) =
    "$(NERSC_DATA_ROOT)/photoObj/301/$(run)/$(camcol)"
nersc_psfield_dir(run::Integer, camcol::Integer) =
    "$(NERSC_DATA_ROOT)/photo/redux/301/$(run)/objcs/$(camcol)"
nersc_photofield_dir(run::Integer) =
    "$(NERSC_DATA_ROOT)/photoObj/301/$(run)"
nersc_frame_dir(run::Integer, camcol::Integer) =
    "$(NERSC_DATA_ROOT)/photoObj/frames/301/$(run)/$(camcol)"
nersc_fpm_dir(run::Integer, camcol::Integer) =
    "$(NERSC_DATA_ROOT)/photo/redux/301/$(run)/objcs/$(camcol)"


# NERSC scratch directories
nersc_field_scratchdir(run::Integer, camcol::Integer, field::Integer) =
    joinpath(ENV["SCRATCH"], "celeste/$(run)/$(camcol)/$(field)")
nersc_photofield_scratchdir(run::Integer, camcol::Integer) =
    joinpath(ENV["SCRATCH"], "celeste/$(run)/$(camcol)")

"""
    nersc_stage_field(run, camcol, field)

Stage all relevant files for the given run, camcol, field to user's SCRATCH
directory. The target locations are given by `nersc_field_scratchdir` and
`nersc_photofield_scratchdir`.
"""
function nersc_stage_field(run::Integer, camcol::Integer, field::Integer)
    # destination directory for all files except photofield.
    dstdir = nersc_field_scratchdir(run, camcol, field)
    isdir(dstdir) || mkpath(dstdir)

    # frame files: uncompress bz2 files
    srcdir = nersc_frame_dir(run, camcol)
    for band in ['u', 'g', 'r', 'i', 'z']
        srcfile = @sprintf("%s/frame-%s-%06d-%d-%04d.fits.bz2",
                           srcdir, band, run, camcol, field)
        dstfile = @sprintf("%s/frame-%s-%06d-%d-%04d.fits",
                           dstdir, band, run, camcol, field)
        if !isfile(dstfile)
            Logging.info("bzcat --keep $srcfile > $dstfile")
            Base.run(pipeline(`bzcat --keep $srcfile`, stdout=dstfile))
        end
    end

    # fpm files
    # It isn't strictly necessary to uncompress these, because FITSIO can handle
    # gzipped files. However, the celeste code assumes the filename ends with
    # ".fit", so we would have to at least change the name. It seems clearer
    # to simply uncompress here.
    srcdir = nersc_fpm_dir(run, camcol)
    dstdir = nersc_field_scratchdir(run, camcol, field)
    isdir(dstdir) || mkpath(dstdir)
    for band in ['u', 'g', 'r', 'i', 'z']
        srcfile = @sprintf("%s/fpM-%06d-%s%d-%04d.fit.gz",
                           srcdir, run, band, camcol, field)
        dstfile = @sprintf("%s/fpM-%06d-%s%d-%04d.fit",
                           dstdir, run, band, camcol, field)
        if !isfile(dstfile)
            Logging.info("gunzip --stdout $srcfile > $dstfile")
            Base.run(pipeline(`gunzip --stdout $srcfile`, stdout=dstfile))
        end
    end

    # photoobj: simply copy
    srcfile = @sprintf("%s/photoObj-%06d-%d-%04d.fits",
                       nersc_photoobj_dir(run, camcol), run, camcol, field)
    dstfile = @sprintf("%s/photoObj-%06d-%d-%04d.fits",
                       nersc_field_scratchdir(run, camcol, field), run,
                       camcol, field)
    isfile(dstfile) || cp(srcfile, dstfile)

    # psField: simply copy
    srcfile = @sprintf("%s/psField-%06d-%d-%04d.fit",
                       nersc_psfield_dir(run, camcol), run, camcol, field)
    dstfile = @sprintf("%s/psField-%06d-%d-%04d.fit",
                       nersc_field_scratchdir(run, camcol, field), run,
                       camcol, field)
    isfile(dstfile) || cp(srcfile, dstfile)

    # photofield: simply copy
    srcfile = @sprintf("%s/photoField-%06d-%d.fits",
                       nersc_photofield_dir(run), run, camcol)
    dstfile = @sprintf("%s/photoField-%06d-%d.fits",
                       nersc_photofield_scratchdir(run, camcol), run, camcol)
    isfile(dstfile) || cp(srcfile, dstfile)
end


"""
Stage all relevant files for the given sky patch to user's SCRATCH.
"""
function stage_box_nersc(ramin, ramax, decmin, decmax)
    fieldids = query_overlapping_fieldids(ramin, ramax, decmin, decmax)
    for (run, camcol, field) in fieldids
        nersc_stage_field(run, camcol, field)
    end
end


"""
Save provided results to a JLD file.
"""
function save_results(outdir, ramin, ramax, decmin, decmax, results)
    fname = @sprintf("%s/celeste-%.4f-%.4f-%.4f-%.4f.jld",
                     outdir, ramin, ramax, decmin, decmax)
    JLD.save(fname, "results", results)
end


"""
NERSC-specific infer function, called from main entry point.
"""
function infer_box_nersc(ramin, ramax, decmin, decmax, outdir;
                         stage::Bool=false,
                         show_timing::Bool=true)
    # Get vector of (run, camcol, field) triplets overlapping this patch
    fieldids = query_overlapping_fieldids(ramin, ramax, decmin, decmax)

    if stage
        for (run, camcol, field) in fieldids
            nersc_stage_field(run, camcol, field)
        end
    end

    # Get relevant directories corresponding to each field.
    frame_dirs = [nersc_field_scratchdir(x[1], x[2], x[3]) for x in fieldids]
    photofield_dirs = [nersc_photofield_scratchdir(x[1], x[2])
                       for x in fieldids]

    # Base.@time hack for distributed environment
    gc_stats = ()
    gc_diff_stats = ()
    elapsed_time = 0.0
    if show_timing
        gc_stats = Base.gc_num()
        elapsed_time = time_ns()
    end

    times = InferTiming()
    if Distributed && dt_nnodes > 1
        divide_and_infer(fieldids, frame_dirs;
                         ra_range=(ramin, ramax),
                         dec_range=(decmin, decmax),
                         fpm_dirs=frame_dirs,
                         psfield_dirs=frame_dirs,
                         photoobj_dirs=frame_dirs,
                         photofield_dirs=photofield_dirs,
                         timing=times,
                         outdir=outdir)
    else
        results = infer(fieldids, frame_dirs;
                        ra_range=(ramin, ramax),
                        dec_range=(decmin, decmax),
                        fpm_dirs=frame_dirs,
                        psfield_dirs=frame_dirs,
                        photoobj_dirs=frame_dirs,
                        photofield_dirs=photofield_dirs,
                        timing=times)

        tic()
        save_results(outdir, ramin, ramax, decmin, decmax, results)
        times.write_results = toq()
    end

    if show_timing
        # Base.@time hack for distributed environment
        elapsed_time = time_ns() - elapsed_time
        gc_diff_stats = Base.GC_Diff(Base.gc_num(), gc_stats)
        time_puts(elapsed_time, gc_diff_stats.allocd, gc_diff_stats.total_time,
                  Base.gc_alloc_count(gc_diff_stats))

        times.num_srcs = max(1, times.num_srcs)
        nputs(dt_nodeid, "timing: read_photoobj=$(times.read_photoobj)")
        nputs(dt_nodeid, "timing: read_img=$(times.read_img)")
        nputs(dt_nodeid, "timing: init_mp=$(times.init_mp)")
        nputs(dt_nodeid, "timing: fit_psf=$(times.fit_psf)")
        nputs(dt_nodeid, "timing: opt_srcs=$(times.opt_srcs)")
        nputs(dt_nodeid, "timing: num_srcs=$(times.num_srcs)")
        nputs(dt_nodeid, "timing: average opt_srcs=$(times.opt_srcs/times.num_srcs)")
        nputs(dt_nodeid, "timing: write_results=$(times.write_results)")
        nputs(dt_nodeid, "timing: wait_done=$(times.wait_done)")
    end
end


"""
NERSC-specific infer function, called from main entry point.
"""
function infer_field_nersc(run::Int, camcol::Int, field::Int,
                           outdir::AbstractString; objid="")
    # ensure that files are staged and set up paths.
    nersc_stage_field(run, camcol, field)
    field_dirs = [nersc_field_scratchdir(run, camcol, field)]
    photofield_dirs = [nersc_photofield_scratchdir(run, camcol)]

    results = infer([(run, camcol, field)], field_dirs;
                    objid=objid,
                    fpm_dirs=field_dirs,
                    psfield_dirs=field_dirs,
                    photoobj_dirs=field_dirs,
                    photofield_dirs=photofield_dirs,
                    primary_initialization=false)

    fname = if objid == ""
        @sprintf "%s/celeste-%06d-%d-%04d.jld" outdir run camcol field
    else
        @sprintf "%s/celeste-objid-%s.jld" outdir objid
    end
    JLD.save(fname, "results", results)
    Logging.debug("infer_field_nersc finished successfully")
end

