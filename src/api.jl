import FITSIO
import JLD
import Logging

using .Model
import .SDSSIO
import .Infer
import .OptimizeElbo


const TILE_WIDTH = 20
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
const wira = 0.025
const widec = 0.025

"""
Timing information.
"""
type InferTiming
    query_fids::Float64
    get_dirs::Float64
    num_infers::Int64
    read_photoobj::Float64
    read_img::Float64
    init_mp::Float64
    fit_psf::Float64
    opt_srcs::Float64
    num_srcs::Int64
    write_results::Float64
    wait_done::Float64

    InferTiming() = new(0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)
end

function add_timing!(i::InferTiming, j::InferTiming)
    i.query_fids = i.query_fids + j.query_fids
    i.get_dirs = i.get_dirs + j.get_dirs
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


"""
Divide the given ra, dec range into sky areas of `wira`x`widec` and
use Dtree to distribute these sky areas to nodes. Within each node
use `infer()` to fit the Celeste model to sources in each sky area.
"""
function divide_and_infer(ra_range::Tuple{Float64, Float64},
                          dec_range::Tuple{Float64, Float64};
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

        # Get vector of (run, camcol, field) triplets overlapping this patch
        tic()
        fieldids = query_overlapping_fieldids(iramin, iramax,
                                              idecmin, idecmax)
        itimes.query_fids = toq()

        # Get relevant directories corresponding to each field.
        tic()
        frame_dirs = query_frame_dirs(fieldids)
        photofield_dirs = query_photofield_dirs(fieldids)
        itimes.get_dirs = toq()

        # run inference for this subarea
        results = infer(fieldids,
                        frame_dirs;
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

        timing.num_infers = timing.num_infers+1
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


function load_images(fieldids, frame_dirs, fpm_dirs, psfield_dirs, photofield_dirs)
    images = TiledImage[]
    image_names = ASCIIString[]
    image_count = 0

    for i in 1:length(fieldids)
        Logging.info("reading field ", fieldids[i])
        run, camcol, field = fieldids[i]
        field_images = SDSSIO.load_field_images(run, camcol, field, frame_dirs[i],
                                             fpm_dir=fpm_dirs[i],
                                             psfield_dir=psfield_dirs[i],
                                             photofield_dir=photofield_dirs[i])
        for b=1:length(field_images)
            image_count += 1
            push!(image_names,
                "$image_count run=$run camcol=$camcol $field=field b=$b")
            tiled_image = TiledImage(field_images[b])
            push!(images, tiled_image)
        end
    end
    gc()

    Logging.debug("Image names:")
    Logging.debug(image_names)

    images
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
               reserve_thread=Ref(false),
               thread_fun=phalse,
               timing=InferTiming())

    Logging.info("Running with $(nthreads()) threads")

    # Read all primary objects in these fields.
    tic()
    duplicate_policy = primary_initialization ? :primary : :first
    catalog = SDSSIO.read_photoobj_files(fieldids, photoobj_dirs,
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
    tic()
    images = load_images(fieldids, frame_dirs, fpm_dirs, psfield_dirs, photofield_dirs)
    timing.read_img = toq()

    reserve_thread[] && thread_fun(reserve_thread)

    Logging.info("finding neighbors")
    tic()
    neighbor_map = Infer.find_neighbors(target_sources, catalog, images)
    Logging.info("neighbors found in ", toq(), " seconds")

    reserve_thread[] && thread_fun(reserve_thread)

    # iterate over sources
    results = Dict{Int, Dict}()
    results_lock = SpinLock()
    tic()
    @threads for ts in 1:length(target_sources)
        tid = threadid()
        if reserve_thread[] && tid == 1
            while reserve_thread[]
                thread_fun(reserve_thread)
                cpu_pause()
            end
        else
            s = target_sources[ts]
            entry = catalog[s]

#            try
                nputs(dt_nodeid, "processing source $s: objid = $(entry.objid)")
                gc()

                t0 = time()
                # TODO: subset images to images_local too.
                vs_opt = Infer.infer_source(images,
                                            catalog[neighbor_map[ts]],
                                            entry)
                runtime = time() - t0

                lock!(results_lock)
                results[entry.thing_id] = Dict(
                             "objid"=>entry.objid,
                             "ra"=>entry.pos[1],
                             "dec"=>entry.pos[2],
                             "vs"=>vs_opt,
                             "runtime"=>runtime)
                unlock!(results_lock)
#            catch ex
#                Logging.err(ex)
#            end
        end
    end
    timing.opt_srcs = toq()
    timing.num_srcs = length(target_sources)

    results
end

