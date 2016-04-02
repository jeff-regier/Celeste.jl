# Functions for interacting with Celeste from the command line.

using DataFrames
import Base.+
import FITSIO
import JLD
using Logging  # just for testing right now

using .Types
import .SDSS
import .SkyImages
import .ModelInit
import .OptimizeElbo

const TILE_WIDTH = 20
const DEFAULT_MAX_ITERS = 50
const MIN_FLUX = 2.0

# Use distributed parallelism (with Dtree)
const Distributed = true
if Distributed && VERSION > v"0.5.0-dev"
    using Dtree
else
    const dt_nodeid = 1
    const dt_nnodes = 1
    DtreeScheduler(n, f) = ()
    initwork(dt) = 0, (1, 0)
    getwork(dt) = 0, (1, 0)
    runtree(dt) = 0
    cpu_pause() = ()
end

# Use threads (on the loop over sources)
const Threaded = true
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
const wira = 0.04
const widec = 0.04

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

function +(i1::InferTiming, i2::InferTiming)
    j = InferTiming()
    j.num_infers = i1.num_infers + i2.num_infers
    j.read_photoobj = i1.read_photoobj + i2.read_photoobj
    j.read_img = i1.read_img + i2.read_img
    j.init_mp = i1.init_mp + i2.init_mp
    j.fit_psf = i1.fit_psf + i2.fit_psf
    j.opt_srcs = i1.opt_srcs + i2.opt_srcs
    j.num_srcs = i1.num_srcs + i2.num_srcs
    j.write_results = i1.write_results + i2.write_results
    j.wait_done = i1.wait_done + i2.wait_done
    return j
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
      Logging.configure(level=OFF)
    elseif level == "DEBUG"
      Logging.configure(level=DEBUG)
    elseif level == "INFO"
      Logging.configure(level=INFO)
    elseif level == "WARNING"
      Logging.configure(level=WARNING)
    elseif level == "ERROR"
      Logging.configure(level=ERROR)
    elseif level == "CRITICAL"
      Logging.configure(level=CRITICAL)
    else
      err("Unknown logging level $(level)")
    end
end


immutable MatchException <: Exception
    msg::ASCIIString
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

    info("reading photoobj catalogs for ", length(fieldids), " fields")

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
        info("field $(fieldids[i]): reading $fname")
        rawcatalogs[i] = SDSSIO.read_photoobj(fname)
    end

    for i in eachindex(fieldids)
        info("field ", fieldids[i], ": ", length(rawcatalogs[i]["objid"]),
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
        info("field ", fieldids[i], ": ", length(rawcatalogs[i]["objid"]),
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
    catalog = SkyImages.convert(Vector{CatalogEntry}, rawcatalog)

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
                          times=InferTiming(),
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
    dt = DtreeScheduler(num_work_items, 0.4)
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
        nputs(dt_nodeid, "running inference for $iramin, $iramax, $idecmin, $idecmax")
        results = infer(fieldids, frame_dirs;
                        ra_range=(iramin, iramax),
                        dec_range=(idecmin, idecmax),
                        fpm_dirs=frame_dirs,
                        psfield_dirs=frame_dirs,
                        photoobj_dirs=frame_dirs,
                        photofield_dirs=photofield_dirs,
                        reserve_thread=rundt,
                        thread_fun=rundtree,
                        times=itimes)
        tic()
        output_results(outdir, iramin, iramax, idecmin, idecmax, results)
        itimes.write_results = toq()

        times += itimes
        rundtree(rundt)
    end
    nputs(dt_nodeid, "out of work")
    tic()
    while rundt[]
        rundtree(rundt)
        cpu_pause()
    end
    finalize(dt)
    times.wait_done = toq()
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
               times=InferTiming())
    # Read all primary objects in these fields.
    tic()
    duplicate_policy = primary_initialization ? :primary : :first
    catalog = read_photoobj_files(fieldids, photoobj_dirs,
                        duplicate_policy=duplicate_policy)
    times.read_photoobj = toq()
    info("$(length(catalog)) primary sources")

    reserve_thread[] && thread_fun(reserve_thread)

    # Filter out low-flux objects in the catalog.
    catalog = filter(entry->(maximum(entry.star_fluxes) >= MIN_FLUX), catalog)
    info("$(length(catalog)) primary sources after MIN_FLUX cut")

    # Filter any object not specified, if an objid is specified
    if objid != ""
        info(catalog[1].objid)
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
        info("reading field ", fieldids[i])
        run, camcol, field = fieldids[i]
        fieldims = SkyImages.read_sdss_field(run, camcol, field, frame_dirs[i],
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
    times.read_img = toq()

    reserve_thread[] && thread_fun(reserve_thread)

    debug("Image names:")
    debug(image_names)

    # initialize tiled images and model parameters for trimming.  We will
    # initialize the psf again before fitting, so we don't do it here.
    info("initializing celeste without PSF fit")
    tic()
    tiled_images = SkyImages.break_blob_into_tiles(images, TILE_WIDTH)
    mp = ModelInit.initialize_model_params(tiled_images, images, catalog,
                                           fit_psf=false)
    times.init_mp = toq()

    reserve_thread[] && thread_fun(reserve_thread)

    # get indicies of all sources relevant to those we're actually
    # interested in, and fit a local PSF for those sources (since we skipped
    # fitting the PSF for the whole catalog above)
    info("fitting PSF for all relevant sources")
    tic()
    ModelInit.fit_object_psfs!(mp, target_sources, images)
    times.fit_psf = toq()

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

                # hack for Julia's GC
                gc_state = ccall(:jl_gc_safe_enter, Int8, ())
                ccall(:jl_gc_safe_leave, Void, (Int8,), gc_state)
            end
        else
            entry = catalog[s]

            try
                nputs(dt_nodeid, "processing source $s: objid = $(entry.objid)")
                tic()

                t0 = time()
                relevant_sources = ModelInit.get_relevant_sources(mp, s)
                mp_source = ModelParams(mp, relevant_sources)
                sa = findfirst(relevant_sources, s)
                mp_source.active_sources = Int[ sa ]
                trimmed_tiled_images =
                  ModelInit.trim_source_tiles(sa, mp_source, tiled_images;
                                              noise_fraction=0.1)
                init_time = time() - t0

                t = toq()
                nputs(dt_nodeid, "trimmed $s in $t secs, optimizing")
                tic()

                t0 = time()
                iter_count, max_f, max_x, result =
                    OptimizeElbo.maximize_f(ElboDeriv.elbo, trimmed_tiled_images,
                                            mp_source;
                                            verbose=false, max_iters=max_iters)
                fit_time = time() - t0

                t = toq()
                nputs(dt_nodeid, "optimized $s in $t secs, writing results")

                lock!(results_lock)
                results[entry.thing_id] = Dict("objid"=>entry.objid,
                                               "ra"=>entry.pos[1],
                                               "dec"=>entry.pos[2],
                                               "vs"=>mp_source.vp[sa],
                                               "init_time"=>init_time,
                                               "fit_time"=>fit_time)
                unlock!(results_lock)
            catch ex
                err(ex)
            end
        end
    end
    times.opt_srcs = toq()
    times.num_srcs = length(target_sources)
    times.num_infers = 1

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
            println("bzcat --keep $srcfile > $dstfile")
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
            println("gunzip --stdout $srcfile > $dstfile")
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
                         times=times,
                         outdir=outdir)
    else
        results = infer(fieldids, frame_dirs;
                        ra_range=(ramin, ramax),
                        dec_range=(decmin, decmax),
                        fpm_dirs=frame_dirs,
                        psfield_dirs=frame_dirs,
                        photoobj_dirs=frame_dirs,
                        photofield_dirs=photofield_dirs,
                        times=times)

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
    debug("infer_field_nersc finished successfully")
end


# -----------------------------------------------------------------------------
# Scoring

"""
mag_to_flux(m)

convert SDSS mags to SDSS flux
"""
mag_to_flux(m::AbstractFloat) = 10.^(0.4 * (22.5 - m))
@vectorize_1arg AbstractFloat mag_to_flux

flux_to_mag(nm::AbstractFloat) = nm > 0 ? 22.5 - 2.5 * log10(nm) : NaN
@vectorize_1arg AbstractFloat flux_to_mag

"""
where(condition, x, y)

Construct a new Array containing elements from `x` where `condition` is true
otherwise elements from `y`.
"""
function where(condition, x, y)
    @assert length(condition) == length(x) == length(y)
    out = similar(x)
    for i=1:length(condition)
        out[i] = condition[i]? x[i]: y[i]
    end
    return out
end


"""
Return distance in pixels using small-distance approximation. Falls
apart at poles and RA boundary.
"""
dist(ra1, dec1, ra2, dec2) = (3600 / 0.396) * (sqrt((dec2 - dec1).^2 +
                                  (cos(dec1) .* (ra2 - ra1)).^2))

"""
match_position(ras, decs, ra, dec, dist)

Return index of first position in `ras`, `decs` that is within a distance
`maxdist` (in degrees) of the target position `ra`, `dec`. If none found,
an exception is raised.
"""
function match_position(ras, decs, ra, dec, maxdist)
    @assert length(ras) == length(decs)
    for i in 1:length(ras)
        dist(ra, dec, ras[i], decs[i]) < maxdist && return i
    end
    throw(MatchException(@sprintf("No source found at %f  %f", ra, dec)))
end


"""
load_s82(fname)

Load Stripe 82 objects into a DataFrame. `fname` should be a FITS file
created by running a CasJobs (skyserver.sdss.org/casjobs/) query
on the Stripe82 database. Run the following query in the \"Stripe82\"
context, then download the table as a FITS file.

```
select
  objid, rerun, run, camcol, field, flags,
  ra, dec, probpsf,
  psfmag_u, psfmag_g, psfmag_r, psfmag_i, psfmag_z,
  devmag_u, devmag_g, devmag_r, devmag_i, devmag_z,
  expmag_u, expmag_g, expmag_r, expmag_i, expmag_z,
  fracdev_r,
  devab_r, expab_r,
  devphi_r, expphi_r,
  devrad_r, exprad_r
into mydb.s82_0_1_0_1
from stripe82.photoobj
where
  run in (106, 206) and
  ra between 0. and 1. and
  dec between 0. and 1.
```
"""
function load_s82(fname)

    # First, simply read the FITS table into a dictionary of arrays.
    f = FITSIO.FITS(fname)
    keys = [:objid, :rerun, :run, :camcol, :field, :flags,
            :ra, :dec, :probpsf,
            :psfmag_u, :psfmag_g, :psfmag_r, :psfmag_i, :psfmag_z,
            :devmag_u, :devmag_g, :devmag_r, :devmag_i, :devmag_z,
            :expmag_u, :expmag_g, :expmag_r, :expmag_i, :expmag_z,
            :fracdev_r,
            :devab_r, :expab_r,
            :devphi_r, :expphi_r,
            :devrad_r, :exprad_r,
            :flags]
    objs = [key=>read(f[2], string(key)) for key in keys]
    close(f)

    usedev = objs[:fracdev_r] .> 0.5  # true=> use dev, false=> use exp

    # Convert to "celeste" style results.
    gal_mag_u = where(usedev, objs[:devmag_u], objs[:expmag_u])
    gal_mag_g = where(usedev, objs[:devmag_g], objs[:expmag_g])
    gal_mag_r = where(usedev, objs[:devmag_r], objs[:expmag_r])
    gal_mag_i = where(usedev, objs[:devmag_i], objs[:expmag_i])
    gal_mag_z = where(usedev, objs[:devmag_z], objs[:expmag_z])

    result = DataFrame()
    result[:objid] = objs[:objid]
    result[:ra] = objs[:ra]
    result[:dec] = objs[:dec]
    result[:is_star] = [x != 0 for x in objs[:probpsf]]
    result[:star_mag_r] = objs[:psfmag_r]
    result[:gal_mag_r] = gal_mag_r

    # star colors
    result[:star_color_ug] = objs[:psfmag_u] .- objs[:psfmag_g]
    result[:star_color_gr] = objs[:psfmag_g] .- objs[:psfmag_r]
    result[:star_color_ri] = objs[:psfmag_r] .- objs[:psfmag_i]
    result[:star_color_iz] = objs[:psfmag_i] .- objs[:psfmag_z]

    # gal colors
    result[:gal_color_ug] = gal_mag_u .- gal_mag_g
    result[:gal_color_gr] = gal_mag_g .- gal_mag_r
    result[:gal_color_ri] = gal_mag_r .- gal_mag_i
    result[:gal_color_iz] = gal_mag_i .- gal_mag_z

    # gal shape -- fracdev
    result[:gal_fracdev] = objs[:fracdev_r]

    # Note that the SDSS photo pipeline doesn't constrain the de Vaucouleur
    # profile parameters and exponential disk parameters (A/B, angle, scale)
    # to be the same, whereas Celeste does. Here, we pick one or the other
    # from SDSS, based on fracdev - we'll get the parameters corresponding
    # to the dominant component. Later, we limit comparison to objects with
    # fracdev close to 0 or 1 to ensure that we're comparing apples to apples.

    result[:gal_ab] = where(usedev, objs[:devab_r], objs[:expab_r])

    # gal effective radius (re)
    re_arcsec = where(usedev, objs[:devrad_r], objs[:exprad_r])
#    re_arcsec = where(usedev, objs[:theta_dev], objs[:theta_exp])
    re_pixel = re_arcsec / 0.396
    result[:gal_scale] = re_pixel

    # gal angle (degrees)
    raw_phi = where(usedev, objs[:devphi_r], objs[:expphi_r])
    result[:gal_angle] = raw_phi - floor(raw_phi / 180) * 180

    return result
end


"""
Convert two fluxes to a color: mag(f1) - mag(f2), assuming the same zeropoint.
Returns NaN if either flux is nonpositive.
"""
function fluxes_to_color(f1::Real, f2::Real)
    (f1 <= 0. || f2 <= 0.) && return NaN
    return -2.5 * log10(f1 / f2)
end
@vectorize_2arg Real fluxes_to_color


"""
load_primary(dir, run, camcol, field)

Load the SDSS photoObj catalog used to initialize celeste, and reformat column
names to match what the rest of the scoring code expects.
"""
function load_primary(dir, run, camcol, field)
    objs = SDSS.load_catalog_df(dir, run, camcol, field)
    usedev = objs[:frac_dev] .> 0.5  # true=> use dev, false=> use exp

    gal_flux_u = where(usedev, objs[:devflux_u], objs[:expflux_u])
    gal_flux_g = where(usedev, objs[:devflux_g], objs[:expflux_g])
    gal_flux_r = where(usedev, objs[:devflux_r], objs[:expflux_r])
    gal_flux_i = where(usedev, objs[:devflux_i], objs[:expflux_i])
    gal_flux_z = where(usedev, objs[:devflux_z], objs[:expflux_z])

    result = DataFrame()
    result[:objid] = objs[:objid]
    result[:ra] = objs[:ra]
    result[:dec] = objs[:dec]
    result[:is_star] = objs[:is_star]
    result[:star_mag_r] = flux_to_mag(objs[:psfflux_r])
    result[:gal_mag_r] = flux_to_mag(gal_flux_r)

    # star colors
    result[:star_color_ug] = fluxes_to_color(objs[:psfflux_u], objs[:psfflux_g])
    result[:star_color_gr] = fluxes_to_color(objs[:psfflux_g], objs[:psfflux_r])
    result[:star_color_ri] = fluxes_to_color(objs[:psfflux_r], objs[:psfflux_i])
    result[:star_color_iz] = fluxes_to_color(objs[:psfflux_i], objs[:psfflux_z])

    # gal colors
    result[:gal_color_ug] = fluxes_to_color(gal_flux_u, gal_flux_g)
    result[:gal_color_gr] = fluxes_to_color(gal_flux_g, gal_flux_r)
    result[:gal_color_ri] = fluxes_to_color(gal_flux_r, gal_flux_i)
    result[:gal_color_iz] = fluxes_to_color(gal_flux_i, gal_flux_z)

    # gal shape -- fracdev
    result[:gal_fracdev] = objs[:frac_dev]

    # gal shape -- axis ratio
    #TODO: filter when 0.5 < frac_dev < .95
    fits_ab = where(usedev, objs[:ab_dev], objs[:ab_exp])
    result[:gal_ab] = fits_ab

    # gal effective radius (re)
    re_arcsec = where(usedev, objs[:theta_dev], objs[:theta_exp])
    re_pixel = re_arcsec / 0.396
    result[:gal_scale] = re_pixel

    # gal angle (degrees)
    raw_phi = where(usedev, objs[:phi_dev], objs[:phi_exp])
    result[:gal_angle] = raw_phi - floor(raw_phi / 180) * 180

    return result
end


"""
This function converts the parameters from Celeste for one light source
to a CatalogEntry. (which can be passed to load_ce!)
It only needs to be called by load_celeste_obj!
"""
function convert(::Type{CatalogEntry}, vs::Vector{Float64}, objid::ASCIIString,
                 thingid::Int)
    function get_fluxes(i::Int)
        ret = Array(Float64, 5)
        ret[3] = exp(vs[ids.r1[i]] + 0.5 * vs[ids.r2[i]])
        ret[4] = ret[3] * exp(vs[ids.c1[3, i]])
        ret[5] = ret[4] * exp(vs[ids.c1[4, i]])
        ret[2] = ret[3] / exp(vs[ids.c1[2, i]])
        ret[1] = ret[2] / exp(vs[ids.c1[1, i]])
        ret
    end

    CatalogEntry(
        vs[ids.u],
        vs[ids.a[1]] > 0.5,
        get_fluxes(1),
        get_fluxes(2),
        vs[ids.e_dev],
        vs[ids.e_axis],
        vs[ids.e_angle],
        vs[ids.e_scale],
        objid,
        thingid)
end

const color_names = ["ug", "gr", "ri", "iz"]

"""
This function loads one catalog entry into row of i of df, a results data
frame.
ce = Catalog Entry, a row of an astronomical catalog
"""
function load_ce!(i::Int, ce::CatalogEntry, df::DataFrame)
    df[i, :ra] = ce.pos[1]
    df[i, :dec] = ce.pos[2]
    df[i, :is_star] = ce.is_star ? 1. : 0.

    for j in 1:2
        s_type = ["star", "gal"][j]
        fluxes = j == 1 ? ce.star_fluxes : ce.gal_fluxes
        df[i, symbol("$(s_type)_mag_r")] = flux_to_mag(fluxes[3])
        for c in 1:4
            cc = symbol("$(s_type)_color_$(color_names[c])")
            cc_sd = symbol("$(s_type)_color_$(color_names[c])_sd")
            if fluxes[c] > 0 && fluxes[c + 1] > 0  # leave as NA otherwise
                df[i, cc] = -2.5log10(fluxes[c] / fluxes[c + 1])
            else
                df[i, cc] = NaN
            end
        end
    end

    df[i, :gal_fracdev] = ce.gal_frac_dev
    df[i, :gal_ab] = ce.gal_ab
    df[i, :gal_angle] = (180/pi)ce.gal_angle
    df[i, :gal_scale] = ce.gal_scale
    df[i, :objid] = ce.objid
end



"""
Convert Celeste results to a dataframe.
"""
function celeste_to_df(results::Dict{Int, Dict})
    # Initialize dataframe
    N = length(results)
    color_col_names = ["color_$cn" for cn in color_names]
    color_sd_col_names = ["color_$(cn)_sd" for cn in color_names]
    col_names = vcat(["objid", "ra", "dec", "is_star", "star_mag_r",
                      "star_mag_r_sd", "gal_mag_r", "gal_mag_r_sd"],
                     ["star_$c" for c in color_col_names],
                     ["star_$c" for c in color_sd_col_names],
                     ["gal_$c" for c in color_col_names],
                     ["gal_$c" for c in color_sd_col_names],
                     ["gal_fracdev", "gal_ab", "gal_angle", "gal_scale"])
    col_symbols = Symbol[symbol(cn) for cn in col_names]
    col_types = Array(DataType, length(col_names))
    fill!(col_types, Float64)
    col_types[1] = ASCIIString
    df = DataFrame(col_types, N)
    names!(df, col_symbols)

    # Fill dataframe row-by-row.
    i = 0
    for (thingid, result) in results
        i += 1
        vs = result["vs"]
        ce = convert(CatalogEntry, vs, result["objid"], thingid)
        load_ce!(i, ce, df)

        df[i, :is_star] = vs[ids.a[1]]

        #TODO: update UQ to mag units not flux. Also, log-normal now, not gamma.
#        for j in 1:2
#            s_type = ["star", "gal"][j]
#            df[i, symbol("$(s_type)_flux_r_sd")] =
#                sqrt(df[i, symbol("$(s_type)_flux_r")]) * vs[ids.r2[j]]
#            for c in 1:4
#                cc_sd = symbol("$(s_type)_color_$(color_names[c])_sd")
#                df[i, cc_sd] = 2.5 * log10(e) * vs[ids.c2[c, j]]
#            end
#        end
    end

    return df
end



"""
Given two results data frame, one containing ground truth (i.e Coadd)
and one containing predictions (i.e., either Primary of Celeste),
compute an a data frame containing each prediction's error.
(It's not an average of the errors, it's each error.)
Let's call the return type of this function an \"error data frame\".
"""
function get_err_df(truth::DataFrame, predicted::DataFrame)
    color_cols = [symbol("color_$cn") for cn in color_names]
    abs_err_cols = [:gal_fracdev, :gal_ab, :gal_scale]
    col_symbols = vcat([:objid, :position, :missed_stars,
                        :missed_gals, :mag_r],
                       color_cols,
                       abs_err_cols,
                       :gal_angle)

    col_types = fill(Float64, length(col_symbols))
    col_types[1] = ASCIIString
    col_types[3] = col_types[4] = Bool
    ret = DataFrame(col_types, size(truth, 1))
    names!(ret, col_symbols)
    ret[:objid] = truth[:objid]

    predicted_gal = predicted[:is_star] .< .5
    true_gal = truth[:is_star] .< .5
    ret[:missed_stars] =  predicted_gal & !(true_gal)
    ret[:missed_gals] =  !predicted_gal & true_gal

    ret[:position] = dist(truth[:ra], truth[:dec],
                          predicted[:ra], predicted[:dec])

    ret[true_gal, :mag_r] =
        abs(truth[true_gal, :gal_mag_r] - predicted[true_gal, :gal_mag_r])
    ret[!true_gal, :mag_r] =
        abs(truth[!true_gal, :star_mag_r] - predicted[!true_gal, :star_mag_r])

    for cn in color_names
        ret[true_gal, symbol("color_$cn")] =
            abs(truth[true_gal, symbol("gal_color_$cn")] -
                predicted[true_gal, symbol("gal_color_$cn")])
        ret[!true_gal, symbol("color_$cn")] =
            abs(truth[!true_gal, symbol("star_color_$cn")] -
                predicted[!true_gal, symbol("star_color_$cn")])
    end

    for n in abs_err_cols
        ret[n] = abs(predicted[n] - truth[n])
    end

    function degrees_to_diff(a, b)
        angle_between = abs(a - b) % 180
        min(angle_between, 180 - angle_between)
    end

    ret[:gal_angle] = degrees_to_diff(truth[:gal_angle], predicted[:gal_angle])

    ret
end


function match_catalogs(fieldid::Tuple{Int, Int, Int},
               results, truthfile, primary_dir)
    # convert Celeste results to a DataFrame.
    celeste_full_df = celeste_to_df(results)
    println("celeste: $(size(celeste_full_df, 1)) objects")

    # load coadd catalog
    coadd_full_df = load_s82(truthfile)
    println("coadd catalog: $(size(coadd_full_df, 1)) objects")

    # find matches in coadd catalog by position
    disttol = 1.0 / 0.396  # 1 arcsec
    good_coadd_indexes = Int[]
    good_celeste_indexes = Int[]
    for i in 1:size(celeste_full_df, 1)
        try
            j = match_position(coadd_full_df[:ra], coadd_full_df[:dec],
                         celeste_full_df[i, :ra], celeste_full_df[i, :dec],
                         disttol)
            push!(good_celeste_indexes, i)
            push!(good_coadd_indexes, j)
        catch y
            isa(y, MatchException) || throw(y)
        end
    end

    celeste_df = celeste_full_df[good_celeste_indexes, :]
    coadd_df = coadd_full_df[good_coadd_indexes, :]

    # load "primary" catalog (the SDSS photoObj catalog used to initialize
    # celeste).
    primary_full_df = load_primary(primary_dir, fieldid...)

    info("primary catalog: $(size(primary_full_df, 1)) objects")

    # match by object id
    good_primary_indexes = Int[findfirst(primary_full_df[:objid], objid)
                   for objid in celeste_df[:objid]]

    # limit primary to matched items
    primary_df = primary_full_df[good_primary_indexes, :]

    # show that all catalogs have same size, and (hopefully)
    # that not too many sources were filtered
    info("matched celeste catalog: $(size(celeste_df, 1)) objects")
    info("matched coadd catalog: $(size(coadd_df, 1)) objects")
    info("matched primary catalog: $(size(primary_df, 1)) objects")

    # ensure that all objects are matched
    if size(primary_df, 1) != size(celeste_df, 1)
        error("catalog mismatch between celeste and primary")
    end

    (celeste_df, primary_df, coadd_df)
end


function get_scores_df(celeste_err, primary_err, coadd_df)
    ttypes = [Symbol, Float64, Float64, Float64, Float64, Int]
    scores_df = DataFrame(ttypes, size(celeste_err, 2) - 1)
    names!(scores_df, [:field, :primary, :celeste, :diff, :diff_sd, :N])

    for i in 1:(size(celeste_err, 2) - 1)
        nm = names(celeste_err)[i + 1]
        nm != :objid || continue

        pe_good = Bool[!isnan(x) for x in primary_err[:, nm]]
        ce_good = Bool[!isnan(x) for x in celeste_err[:, nm]]
        good_row = pe_good & ce_good
        if string(nm)[1:5] == "star_"
            good_row &= (coadd_df[:is_star] .> 0.5)
        elseif string(nm)[1:4] == "gal_"
            good_row &= (coadd_df[:is_star] .< 0.5)
            if nm in [:gal_ab, :gal_scale, :gal_angle, :gal_fracdev]
                good_row &= !(0.05 .< coadd_df[:gal_fracdev] .< 0.95)
            end
            if nm == :gal_angle
                good_row &= coadd_df[:gal_ab] .< .6
            end
        end

        scores_df[i, :field] = nm
        N_good = sum(good_row)
        scores_df[i, :N] = N_good
        N_good > 0 || continue

        scores_df[i, :primary] = mean(primary_err[good_row, nm])
        scores_df[i, :celeste] = mean(celeste_err[good_row, nm])

        diffs = primary_err[good_row, nm] .- celeste_err[good_row, nm]
        scores_df[i, :diff] = mean(diffs)

        # compute the difference in error rates between celeste and primary
        # if we have enough data to get confidence intervals
        N_good > 1 || continue
        abs_errs = Float64[abs(x) for x in diffs]
        scores_df[i, :diff_sd] = std(abs_errs) / sqrt(N_good)
    end

    scores_df
end


function score_field(fieldid::Tuple{Int, Int, Int},
               results, truthfile, primary_dir)
    (celeste_df, primary_df, coadd_df) = match_catalogs(fieldid,
                                results, truthfile, primary_dir)
    # difference between celeste and coadd
    celeste_err = get_err_df(coadd_df, celeste_df)
    primary_err = get_err_df(coadd_df, primary_df)

    JLD.save("df.jld", "celeste_df", celeste_df,
                       "primary_df", primary_df,
                       "coadd_df", coadd_df,
                       "celeste_err", celeste_err,
                       "primary_err", primary_err)

    # create scores
    get_scores_df(celeste_err, primary_err, coadd_df)
end


"""
Score the celeste results for a particular field
"""
function score_field_nersc(run, camcol, field, resultdir, truthfile)
    fieldid = (run, camcol, field)
    fname = @sprintf "%s/celeste-%06d-%d-%04d.jld" resultdir run camcol field
    results = JLD.load(fname, "results")
    primary_dir = nersc_photoobj_dir(run, camcol)

    println( score_field(fieldid, results, truthfile, primary_dir) )
end


"""
Display results from Celeste, Primary, and Coadd for a particular object
"""
function score_object_nersc(run, camcol, field, objid, resultdir, truthfile)
    fieldid = (run, camcol, field)
    fname = @sprintf "%s/celeste-objid-%s.jld" resultdir objid
    results = JLD.load(fname, "results")
    primary_dir = nersc_photoobj_dir(run, camcol)


    (celeste_df, primary_df, coadd_df) = match_catalogs(fieldid,
                                results, truthfile, primary_dir)

    println("\n\nceleste results:\n")
    println(celeste_df)
    println("\n\nprimary results:\n")
    println(primary_df)
    println("\n\ncoadd results:\n")
    println(coadd_df)
end
