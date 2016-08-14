import FITSIO
import JLD

import .Log
using .Model
import .SDSSIO
import .Infer


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
    lock(l) = ()
    unlock(l) = ()
end


# A workitem is of this ra / dec size
const wira = 0.025
const widec = 0.025


"""
Timing information.
"""
type InferTiming
    query_fids::Float64
    num_infers::Int64
    read_photoobj::Float64
    read_img::Float64
    fit_psf::Float64
    opt_srcs::Float64
    num_srcs::Int64
    write_results::Float64
    wait_done::Float64

    InferTiming() = new(0.0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)
end


function add_timing!(i::InferTiming, j::InferTiming)
    i.query_fids = i.query_fids + j.query_fids
    i.num_infers = i.num_infers + j.num_infers
    i.read_photoobj = i.read_photoobj + j.read_photoobj
    i.read_img = i.read_img + j.read_img
    i.fit_psf = i.fit_psf + j.fit_psf
    i.opt_srcs = i.opt_srcs + j.opt_srcs
    i.num_srcs = i.num_srcs + j.num_srcs
    i.write_results = i.write_results + j.write_results
    i.wait_done = i.wait_done + j.wait_done
end


immutable BoundingBox
    ramin::Float64
    ramax::Float64
    decmin::Float64
    decmax::Float64
end


immutable FieldExtent
    run::Int64
    camcol::Int64
    field::Int64
    box::BoundingBox
end


"""
Given a BoundingBox that is to be divided into `nra` x `ndec` subareas,
return the `i`th subarea. `i` is a linear index between 1 and
`nra * ndec`.

This function assumes a cartesian (rather than spherical) coordinate system!
"""
function divide_skyarea(box, nra, ndec, i)
    global wira, widec
    ix, iy = ind2sub((nra, ndec), i)

    return (box.ramin + (ix - 1) * wira,
            min(box.ramin + ix * wira, box.ramax),
            box.decmin + (iy - 1) * widec,
            min(box.decmin + iy * widec, box.decmax))
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
Divide `N` into `np` parts as evenly as possible, returning `my` part as
a (first, last) tuple.
"""
function divparts(N, np, my)
    len, rem = divrem(N, np)
    if len == 0
        if my > rem
            return 1, 0
        end
        len, rem = 1, 0
    end
    # compute my part
    f = 1 + ((my-1) * len)
    l = f + len - 1
    # distribute remaining evenly
    if rem > 0
        if my <= rem
            f = f + (my-1)
            l = l + my
        else
            f = f + rem
            l = l + rem
        end
    end
    return f, l
end


"""
Divide the given ra, dec range into sky areas of `wira`x`widec` and
use Dtree to distribute these sky areas to nodes. Within each node
use `infer()` to fit the Celeste model to sources in each sky area.
"""
function divide_and_infer(box::BoundingBox,
                          stagedir::String,
                          timing=InferTiming(),
                          outdir=".",
                          output_results=save_results)
    if dt_nodeid == 1
        nputs(dt_nodeid, "running on $dt_nnodes nodes")
    end

    # how many `wira` X `widec` sky areas (work items)?
    global wira, widec
    nra = ceil(Int64, (box.ramax - box.ramin) / wira)
    ndec = ceil(Int64, (box.decmax - box.decmin) / widec)

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
        box = BoundingBox(iramin, iramax, idecmin, idecmax)
        fieldids = get_overlapping_fieldids(box, stagedir)
        itimes.query_fids = toq()

        # run inference for this subarea
        results = infer(fieldids;
                        box=BoundingBox(iramin, iramax, idecmin, idecmax),
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
    end
    finalize(dt)
    timing.wait_done = toq()
end


function load_images(fieldids, stagedir)
    images = TiledImage[]
    image_names = String[]
    image_count = 0

    for i in 1:length(fieldids)
        Log.info("reading field $(fieldids[i])")
        run, camcol, field = fieldids[i]
        field_images = SDSSIO.load_field_images(run, camcol, field, stagedir)
        for b=1:length(field_images)
            image_count += 1
            push!(image_names,
                "$image_count run=$run camcol=$camcol $field=field b=$b")
            tiled_image = TiledImage(field_images[b])
            push!(images, tiled_image)
        end
    end
    gc()

    Log.debug("Image names:")
    Log.debug(string(image_names))

    images
end


"""
Fit the Celeste model to sources in a given ra, dec range,
based on data from specified fields

- fieldids: Array of run, camcol, field triplets that the source occurs in.
- box: a bounding box specifying a region of sky

Returns:

- Dictionary of results, keyed by SDSS thing_id.
"""
function infer(fieldids::Vector{Tuple{Int, Int, Int}},
               stagedir::String;
               objid="",
               box=BoundingBox(-1000., 1000., -1000., 1000.),
               primary_initialization=true,
               reserve_thread=Ref(false),
               thread_fun=phalse,
               timing=InferTiming())
    nprocthreads = nthreads()
    if reserve_thread[]
        nprocthreads = nprocthreads-1
    end
    Log.info("Running with $(nprocthreads) threads")

    # Read all primary objects in these fields.
    tic()
    duplicate_policy = primary_initialization ? :primary : :first
    catalog = SDSSIO.read_photoobj_files(fieldids, stagedir,
                        duplicate_policy=duplicate_policy)
    timing.read_photoobj = toq()
    Log.info("$(length(catalog)) primary sources")

    reserve_thread[] && thread_fun(reserve_thread)

    # Filter out low-flux objects in the catalog.
    catalog = filter(entry->(maximum(entry.star_fluxes) >= MIN_FLUX), catalog)
    Log.info("$(length(catalog)) primary sources after MIN_FLUX cut")

    # Filter any object not specified, if an objid is specified
    if objid != ""
        Log.info(catalog[1].objid)
        catalog = filter(entry->(entry.objid == objid), catalog)
    end

    # Get indicies of entries in the  RA/Dec range of interest.
    entry_in_range = entry->((box.ramin < entry.pos[1] < box.ramax) &&
                             (box.decmin < entry.pos[2] < box.decmax))
    target_sources = find(entry_in_range, catalog)

    nputs(dt_nodeid, string("processing $(length(target_sources)) sources in ",
          "$(box.ramin), $(box.ramax), $(box.decmin), $(box.decmax)"))

    # If there are no objects of interest, return early.
    if length(target_sources) == 0
        return Dict{Int, Dict}()
    end

    # TODO: make `target_sources` a 1D GlobalArray

    reserve_thread[] && thread_fun(reserve_thread)

    # Read in images for all (run, camcol, field).
    tic()

    # TODO: make `image_map` a 3D GlobalArray
    max_run = maximum([f[1] for f in fieldids])
    max_camcol = maximum([f[2] for f in fieldids])
    max_field = maximum([f[3] for f in fieldids])
    image_map = Array(Int64, max_run, max_camcol, max_field)
    fill!(image_map, 0)

    # TODO: make `images` a 1D GlobalArray
    images = load_images(fieldids, stagedir)
    timing.read_img = toq()

    reserve_thread[] && thread_fun(reserve_thread)

    Log.info("finding neighbors")
    tic()
    neighbor_map = Infer.find_neighbors(target_sources, catalog, images)
    Log.info("neighbors found in $(toq()) seconds")

    reserve_thread[] && thread_fun(reserve_thread)

    # iterate over sources
    results = Dict{Int, Dict}()
    results_lock = SpinLock()
    function process_sources()
        tid = threadid()

        if reserve_thread[] && tid == 1
            while reserve_thread[]
                thread_fun(reserve_thread)
                cpu_pause()
            end
        else
            # divide loop iterations among threads
            f, l = divparts(length(target_sources), nprocthreads, tid)
            for ts = f:l
                s = target_sources[ts]
                entry = catalog[s]

#                try
                    nputs(dt_nodeid, "processing source $s: objid = $(entry.objid)")
                    gc()

                    t0 = time()
                    # TODO: subset images to images_local too.
                    vs_opt = Infer.infer_source(images,
                                                catalog[neighbor_map[ts]],
                                                entry)
                    runtime = time() - t0

                    lock(results_lock)
                    results[entry.thing_id] = Dict(
                                 "objid"=>entry.objid,
                                 "ra"=>entry.pos[1],
                                 "dec"=>entry.pos[2],
                                 "vs"=>vs_opt,
                                 "runtime"=>runtime)
                    unlock(results_lock)
#                catch ex
#                    Log.err(ex)
#                end
            end
        end
    end

    tic()
    ccall(:jl_threading_run, Void, (Any,), Core.svec(process_sources))
    timing.opt_srcs = toq()
    timing.num_srcs = length(target_sources)

    results
end


"""
Query the SDSS database for all fields that overlap the given RA, Dec range.
"""
function get_overlapping_fields(query::BoundingBox, stagedir::String)
    f = FITSIO.FITS("$stagedir/field_extents.fits")

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

    ret = FieldExtent[]

    # The ramin, ramax, etc is a bit unintuitive because we're looking
    # for any overlap.
    for i in eachindex(all_ramin)
        if (all_ramax[i] > query.ramin && all_ramin[i] < query.ramax &&
                all_decmax[i] > query.decmin && all_decmin[i] < query.decmax)
            cur_box = BoundingBox(
                all_ramin[i],
                all_ramax[i],
                all_decmin[i],
                all_decmax[i])
            cur_fe = FieldExtent(
                all_run[i],
                all_camcol[i],
                all_field[i],
                cur_box)
            push!(ret, cur_fe)
        end
    end

    return ret
end


"""
Like `get_overlapping_fields`, but return a Vector of
(run, camcol, field) triplets.
"""
function get_overlapping_fieldids(query::BoundingBox, stagedir::String)
    fes = get_overlapping_fields(query, stagedir)
    Tuple{Int, Int, Int}[(fe.run, fe.camcol, fe.field) for fe in fes]
end


"""
called from main entry point for inference for one field
(used for accuracy assessment, infer-box is the primary inference
entry point)
"""
function infer_field(run::Int, camcol::Int, field::Int,
                     stagedir::String,
                     outdir::String;
                     objid="")
    results = infer([(run, camcol, field)], stagedir;
                    objid=objid,
                    primary_initialization=false)

    fname = if objid == ""
        @sprintf "%s/celeste-%06d-%d-%04d.jld" outdir run camcol field
    else
        @sprintf "%s/celeste-objid-%s.jld" outdir objid
    end
    JLD.save(fname, "results", results)
    Log.debug("infer_field finished successfully")
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
function infer_box(box::BoundingBox, stagedir::String, outdir::String)
    # Base.@time hack for distributed environment
    gc_stats = ()
    gc_diff_stats = ()
    elapsed_time = 0.0
    gc_stats = Base.gc_num()
    elapsed_time = time_ns()

    times = InferTiming()
    if dt_nnodes > 1
        divide_and_infer(box,
                         stagedir,
                         timing=times,
                         outdir=outdir)
    else
        # Get vector of (run, camcol, field) triplets overlapping this patch
        tic()
        fieldids = get_overlapping_fieldids(box, stagedir)
        times.query_fids = toq()

        results = infer(fieldids, stagedir; box=box, timing=times)

        tic()
        save_results(outdir, box, results)
        times.write_results = toq()
    end

    # Base.@time hack for distributed environment
    elapsed_time = time_ns() - elapsed_time
    gc_diff_stats = Base.GC_Diff(Base.gc_num(), gc_stats)
    time_puts(elapsed_time, gc_diff_stats.allocd, gc_diff_stats.total_time,
              Base.gc_alloc_count(gc_diff_stats))

    times.num_srcs = max(1, times.num_srcs)
    nputs(dt_nodeid, "timing: query_fids=$(times.query_fids)")
    nputs(dt_nodeid, "timing: num_infers=$(times.num_infers)")
    nputs(dt_nodeid, "timing: read_photoobj=$(times.read_photoobj)")
    nputs(dt_nodeid, "timing: read_img=$(times.read_img)")
    nputs(dt_nodeid, "timing: fit_psf=$(times.fit_psf)")
    nputs(dt_nodeid, "timing: opt_srcs=$(times.opt_srcs)")
    nputs(dt_nodeid, "timing: num_srcs=$(times.num_srcs)")
    nputs(dt_nodeid, "timing: average opt_srcs=$(times.opt_srcs/times.num_srcs)")
    nputs(dt_nodeid, "timing: write_results=$(times.write_results)")
    nputs(dt_nodeid, "timing: wait_done=$(times.wait_done)")
end

