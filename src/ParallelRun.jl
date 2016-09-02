module ParallelRun

import FITSIO
import JLD

import ..Log
using ..Model
import ..SDSSIO
import ..Infer
import ..SDSSIO: RunCamcolField


using Base.Threads
using Garbo

const TILE_WIDTH = 20
const MIN_FLUX = 2.0

# set this to false to use source-division parallelism
const SKY_DIVISION_PARALLELISM=true

# In sky-division parallelism, a workitem is of this ra / dec size
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


function BoundingBox(ramin::String, ramax::String, decmin::String, decmax::String)
    BoundingBox(parse(Float64, ramin),
                parse(Float64, ramax),
                parse(Float64, decmin),
                parse(Float64, decmax))
end


@inline nputs(nid, s) = ccall(:puts, Cint, (Ptr{Int8},), string("[$nid] ", s))
@inline phalse(b) = b[] = false


include("source_division_inference.jl")


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
    nputs(nodeid, s)
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
use `one_node_infer()` to fit the Celeste model to sources in each sky area.
"""
function divide_sky_and_infer(
                box::BoundingBox,
                stagedir::String;
                timing=InferTiming(),
                outdir=".")
    if nodeid == 1
        nputs(nodeid, "running on $nnodes nodes")
    end

    # how many `wira` X `widec` sky areas (work items)?
    global wira, widec
    nra = ceil(Int64, (box.ramax - box.ramin) / wira)
    ndec = ceil(Int64, (box.decmax - box.decmin) / widec)

    num_work_items = nra * ndec
    each = ceil(Int64, num_work_items / nnodes)

    if nodeid == 1
        nputs(nodeid, "work item dimensions: $wira X $widec")
        nputs(nodeid, "$num_work_items work items, ~$each per node")
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
    nputs(nodeid, "initially $ni work items ($ci to $li)")
    itimes = InferTiming()
    while ni > 0
        li == 0 && break
        if ci > li
            nputs(nodeid, "consumed allocation (last was $li)")
            ni, (ci, li) = getwork(dt)
            nputs(nodeid, "got $ni work items ($ci to $li)")
            continue
        end
        item = ci
        ci = ci + 1

        # map item to subarea
        iramin, iramax, idecmin, idecmax = divide_skyarea(box, nra, ndec, item)

        # Get vector of (run, camcol, field) triplets overlapping this patch
        tic()
        box = BoundingBox(iramin, iramax, idecmin, idecmax)
        rcfs = get_overlapping_fields(box, stagedir)
        itimes.query_fids = toq()

        # run inference for this subarea
        results = one_node_infer(rcfs, stagedir;
                        box=BoundingBox(iramin, iramax, idecmin, idecmax),
                        reserve_thread=rundt,
                        thread_fun=rundtree,
                        timing=itimes)
        tic()
        save_results(outdir, iramin, iramax, idecmin, idecmax, results)
        itimes.write_results = toq()

        timing.num_infers = timing.num_infers+1
        add_timing!(timing, itimes)
        rundtree(rundt)
    end
    nputs(nodeid, "out of work")
    tic()
    while rundt[]
        rundtree(rundt)
    end
    finalize(dt)
    timing.wait_done = toq()
end


function load_images(rcfs, stagedir)
    images = TiledImage[]
    image_names = String[]
    image_count = 0

    for i in 1:length(rcfs)
        Log.info("reading field $(rcfs[i])")
        rcf = rcfs[i]
        field_images = SDSSIO.load_field_images(rcf, stagedir)
        for b=1:length(field_images)
            image_count += 1
            push!(image_names,
                "$image_count run=$(rcf.run) camcol=$(rcf.camcol) field=$(rcf.field) b=$b")
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
Use mulitple threads on one node to 
fit the Celeste model to sources in a given bounding box.

- rcfs: Array of run, camcol, field triplets that the source occurs in.
- box: a bounding box specifying a region of sky

Returns:

- Dictionary of results, keyed by SDSS thing_id.
"""
function one_node_infer(
               rcfs::Vector{RunCamcolField},
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
    catalog = SDSSIO.read_photoobj_files(rcfs, stagedir,
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

    nputs(nodeid, string("processing $(length(target_sources)) sources in ",
          "$(box.ramin), $(box.ramax), $(box.decmin), $(box.decmax)"))

    # If there are no objects of interest, return early.
    if length(target_sources) == 0
        return Dict{Int, Dict}()
    end

    reserve_thread[] && thread_fun(reserve_thread)

    # Read in images for all (run, camcol, field).
    tic()

    images = load_images(rcfs, stagedir)
    timing.read_img = toq()

    reserve_thread[] && thread_fun(reserve_thread)

    Log.info("finding neighbors")
    tic()
    neighbor_map = Infer.find_neighbors(target_sources, catalog, images)
    Log.info("neighbors found in $(toq()) seconds")

    reserve_thread[] && thread_fun(reserve_thread)

    # iterate over sources
    curr_source = 1
    last_source = length(target_sources)
    sources_lock = SpinLock()
    results = Dict[]
    results_lock = SpinLock()
    function process_sources()
        tid = threadid()

        if reserve_thread[] && tid == 1
            while reserve_thread[]
                thread_fun(reserve_thread)
                cpu_pause()
            end
        else
            gc_enable(false)
            while true
                lock(sources_lock)
                ts = curr_source
                curr_source += 1
                unlock(sources_lock)

                if ts > last_source
                    break
                end
#                try
                    s = target_sources[ts]
                    entry = catalog[s]
                    nputs(nodeid, "processing source $s: objid = $(entry.objid)")

                    t0 = time()
                    # TODO: subset images to images_local too.
                    vs_opt = Infer.infer_source(images,
                                                catalog[neighbor_map[ts]],
                                                entry)
                    runtime = time() - t0
#                catch ex
#                    Log.error(ex)
#                end

                lock(results_lock)
                push!(results, Dict(
                    "thing_id"=>entry.thing_id,
                    "objid"=>entry.objid,
                    "ra"=>entry.pos[1],
                    "dec"=>entry.pos[2],
                    "vs"=>vs_opt,
                    "runtime"=>runtime))
                unlock(results_lock)
            end
        end
        gc_enable(true)
    end

    tic()
    ccall(:jl_threading_run, Void, (Any,), Core.svec(process_sources))
    ccall(:jl_threading_profile, Void, ())
    timing.opt_srcs = toq()
    timing.num_srcs = length(target_sources)

    results
end


"""
Query the SDSS database for all fields that overlap the given RA, Dec range.
"""
function get_overlapping_field_extents(query::BoundingBox, stagedir::String)
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


"""
Like `get_overlapping_fields`, but return a Vector of
(run, camcol, field) triplets.
"""
function get_overlapping_fields(query::BoundingBox, stagedir::String)
    fes = get_overlapping_field_extents(query, stagedir)
    [fe[1] for fe in fes]
end


"""
called from main entry point for inference for one field
(used for accuracy assessment, infer-box is the primary inference
entry point)
"""
function infer_field(rcf::RunCamcolField,
                     stagedir::String,
                     outdir::String;
                     objid="")
    results = one_node_infer([rcf,], stagedir; objid=objid, primary_initialization=false)
    fname = if objid == ""
        @sprintf "%s/celeste-%06d-%d-%04d.jld" outdir rcf.run rcf.camcol rcf.field
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

save_results(outdir, box, results) =
    save_results(outdir, box.ramin, box.ramax, box.decmin, box.decmax, results)


"""
called from main entry point.
"""
function infer_box(box::BoundingBox, stagedir::String, outdir::String)
    # Base.@time hack for distributed environment
    gc_stats = ()
    gc_diff_stats = ()
    elapsed_time = 0.0
    gc_stats = Base.gc_num()
    elapsed_time = time_ns()

    times = InferTiming()
    if !SKY_DIVISION_PARALLELISM
        Log.debug("source division parallelism")
        divide_sources_and_infer(box, stagedir; timing=times, outdir=outdir)
    elseif nnodes > 1
        divide_sky_and_infer(box, stagedir; timing=times, outdir=outdir)
    else
        Log.debug("multithreaded parallelism only")
        tic()
        # Get vector of (run, camcol, field) triplets overlapping this patch
        rcfs = get_overlapping_fields(box, stagedir)
        times.query_fids = toq()

        results = one_node_infer(rcfs, stagedir; box=box, timing=times)

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
    nputs(nodeid, "timing: query_fids=$(times.query_fids)")
    nputs(nodeid, "timing: num_infers=$(times.num_infers)")
    nputs(nodeid, "timing: read_photoobj=$(times.read_photoobj)")
    nputs(nodeid, "timing: read_img=$(times.read_img)")
    nputs(nodeid, "timing: fit_psf=$(times.fit_psf)")
    nputs(nodeid, "timing: opt_srcs=$(times.opt_srcs)")
    nputs(nodeid, "timing: num_srcs=$(times.num_srcs)")
    nputs(nodeid, "timing: average opt_srcs=$(times.opt_srcs/times.num_srcs)")
    nputs(nodeid, "timing: write_results=$(times.write_results)")
    nputs(nodeid, "timing: wait_done=$(times.wait_done)")
end

end
