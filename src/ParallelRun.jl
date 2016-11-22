module ParallelRun

import FITSIO
import JLD

import ..Log
using ..Model
import ..SDSSIO
import ..Infer
import ..SDSSIO: RunCamcolField
import ..PSF

import ..DeterministicVI: infer_source


include("cyclades.jl")

#set this to false to use source-division parallelism
const SKY_DIVISION_PARALLELISM=true

const MIN_FLUX = 2.0

# In production mode, rather the development mode, always catch exceptions
const is_production_run = haskey(ENV, "CELESTE_PROD") &&
                          ENV["CELESTE_PROD"] != ""

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
use `one_node_infer()` to fit the Celeste model to sources in each sky area.
"""
function divide_sky_and_infer(
                box::BoundingBox,
                stagedir::String;
                timing=InferTiming(),
                outdir=".")
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
        iramin, iramax, idecmin, idecmax = divide_skyarea(box, nra, ndec, item)

        # Get vector of (run, camcol, field) triplets overlapping this patch
        tic()
        box = BoundingBox(iramin, iramax, idecmin, idecmax)
        rcfs = get_overlapping_fields(box, stagedir)
        itimes.query_fids = toq()

        # run inference for this subarea
        results, obj_value = one_node_infer(rcfs, stagedir;
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
    nputs(dt_nodeid, "out of work")
    tic()
    while rundt[]
        rundtree(rundt)
    end
    finalize(dt)
    timing.wait_done = toq()
end


function infer_init(rcfs::Vector{RunCamcolField},
                    stagedir::String;
                    objid="",
                    box=BoundingBox(-1000., 1000., -1000., 1000.),
                    target_rcfs=RunCamcolField[],
                    primary_initialization=true)
    # Read all primary objects in these fields.
    duplicate_policy = primary_initialization ? :primary : :first
    catalog = SDSSIO.read_photoobj_files(rcfs, stagedir,
                        duplicate_policy=duplicate_policy)
    Log.info("$(length(catalog)) primary sources")

    # Filter out low-flux objects in the catalog.
    catalog = filter(entry->(maximum(entry.star_fluxes) >= MIN_FLUX), catalog)
    Log.info("$(length(catalog)) primary sources after MIN_FLUX cut")

    # Get indicies of entries in the RA/Dec range of interest.
    entry_in_range = entry->((box.ramin < entry.pos[1] < box.ramax) &&
                             (box.decmin < entry.pos[2] < box.decmax))
    target_sources = find(entry_in_range, catalog)

    Log.info(string("Found $(length(target_sources)) target sources in ",
          "$(box.ramin), $(box.ramax), $(box.decmin), $(box.decmax)"))

    # For infer-box.jl, target sources are everything in the box.
    # For infer-rcf.jl, target sources are primary detections in the target rcf.
    if !isempty(target_rcfs)
        target_entries = SDSSIO.read_photoobj_files(target_rcfs,
                                                    stagedir,
                                                    duplicate_policy=:primary)
        target_ids = Set([entry.objid for entry in target_entries])
        target_sources = filter(ts->(catalog[ts].objid in target_ids), target_sources)
        Log.info("$(length(target_sources)) target light sources after target_rcf cut")
    end

    # Filter any object not specified, if an objid is specified
    if objid != ""
        target_sources = filter(ts->(catalog[ts].objid == objid), target_sources)
        Log.info("$(length(target_sources)) target light sources after objid cut")
    end

    # If there are no objects of interest, return early.
    if length(target_sources) == 0
        images = Image[]
        neighbor_map = Vector{Int}[]
    else
        # Read in images for all (run, camcol, field).
        images = SDSSIO.load_field_images(rcfs, stagedir)

        tic()
        neighbor_map = Infer.find_neighbors(target_sources, catalog, images)
        Log.info("neighbors found in $(toq()) seconds")
    end

    return catalog, target_sources, neighbor_map, images
end

"""
Use mulitple threads on one node to
fit the Celeste model to sources in a given bounding box.

- rcfs: Array of run, camcol, field triplets that the source occurs in.
- box: a bounding box specifying a region of sky

Returns:

- Dictionary of results, keyed by SDSS thing_id.
"""
function one_node_infer(rcfs::Vector{RunCamcolField},
                        stagedir::String;
                        joint_infer=false,
                        joint_infer_n_iters=10,
                        objid="",
                        box=BoundingBox(-1000., 1000., -1000., 1000.),
                        target_rcfs=RunCamcolField[],
                        primary_initialization=true,
                        reserve_thread=Ref(false),
                        thread_fun=phalse,
                        timing=InferTiming())
    # ctni = (catalog, target_sources, neighbor_map, images)
    ctni = infer_init(rcfs, stagedir;
                      objid=objid,
                      box=box,
                      target_rcfs=target_rcfs,
                      primary_initialization=primary_initialization)

    Log.info("Running with $(nthreads()) threads")

    # NB: All I/O happens above in `infer_init()`. The methods below don't
    # touch disk.
    if joint_infer
        return one_node_joint_infer(ctni...;
                                    n_iters=joint_infer_n_iters,
                                    objid=objid)
    else
        return one_node_single_infer(ctni...;
                                     reserve_thread=reserve_thread,
                                     thread_fun=thread_fun,
                                     timing=timing)
    end
end


function one_node_single_infer(catalog::Vector{CatalogEntry},
                               target_sources::Vector{Int},
                               neighbor_map::Vector{Vector{Int}},
                               images::Vector{Image};
                               reserve_thread=Ref(false),
                               thread_fun=phalse,
                               timing=InferTiming())
    obj_values = Array{Float64}(length(target_sources))
    curr_source = 1
    last_source = length(target_sources)
    sources_lock = SpinLock()
    results = Dict[]
    results_lock = SpinLock()

    # iterate over sources
    function process_sources()
        tid = threadid()

        if reserve_thread[] && tid == 1
            while reserve_thread[]
                thread_fun(reserve_thread)
                cpu_pause()
            end
        else
            while true
                lock(sources_lock)
                ts = curr_source
                curr_source += 1
                unlock(sources_lock)
                if ts > last_source
                    break
                end

                try
                    s = target_sources[ts]
                    entry = catalog[s]
                    Log.info("processing source $s: objid = $(entry.objid)")

                    # could subset images to images_local here too.
                    neighbors = catalog[neighbor_map[ts]]

                    t0 = time()
                    vs_opt, obj_value = infer_source(images, neighbors, entry)
                    runtime = time() - t0

                    obj_values[ts] = obj_value

                    result = Dict(
                        "thing_id"=>entry.thing_id,
                        "objid"=>entry.objid,
                        "ra"=>entry.pos[1],
                        "dec"=>entry.pos[2],
                        "vs"=>vs_opt,
                        "runtime"=>runtime)
                    lock(results_lock)
                    push!(results, result)
                    unlock(results_lock)

                    rt1 = round(runtime, 1)
                    Log.info("objid $(entry.objid) took $rt1 seconds")
                    Log.info("========================")
                catch ex
                    if is_production_run || nthreads() > 1
                        Log.error(string(ex))
                    else
                        rethrow(ex)
                    end
                end
            end
        end
    end

    tic()
    if nthreads() == 1
        process_sources()
    else
        ccall(:jl_threading_run, Void, (Any,), Core.svec(process_sources))
        ccall(:jl_threading_profile, Void, ())
    end
    timing.opt_srcs = toq()
    timing.num_srcs = length(target_sources)

    results, obj_values
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
    # Here `one_node_infer` is called just with a single rcf, even though
    # other rcfs may overlap with this one. That's because this function is
    # just for testing on stripe 82: in practice we always use all relevent
    # data to make inferences.
    results, obj_value = one_node_infer([rcf,],
                                        stagedir;
                                        objid=objid,
                                        primary_initialization=false)
    fname = if objid == ""
        @sprintf "%s/celeste-%06d-%d-%04d.jld" outdir rcf.run rcf.camcol rcf.field
    else
        @sprintf "%s/celeste-objid-%s.jld" outdir objid
    end
    JLD.save(fname, "results", results)
    Log.debug("infer_field finished successfully")
end


immutable UnknownRCF <: Exception
    rcf::RunCamcolField
end


"""
This function is called directly from the `bin` directory.
It optimizes all the primary detections in the specified run-camcol-field,
using essentially all relevant images. Typically relevant image data includes
run-camcol-field's in addition to the specified one, that overlap
with the specified run-camcol-field. I say "essentially" because
some large light sources may contribute photons to images that do
not overlap with the rcf containig the center of the light source's
primary detection, and these images are excluded. This is more of a
feature than a bug because we need to limit the amount of computation
per light souce.
"""
function infer_rcf(rcf::RunCamcolField,
                   stagedir::String,
                   outdir::String;
                   objid="")
    whole_sky = BoundingBox(-999, 999, -999, 999)
    all_fes = get_overlapping_field_extents(whole_sky, stagedir)
    this_fe = filter(fe->(fe[1] == rcf), all_fes)

    @assert(length(this_fe) <= 1)
    if isempty(this_fe)
        throw(UnknownRCF(rcf))
    end

    rcf_bounds = this_fe[1][2]
    overlapping_rcfs = get_overlapping_fields(rcf_bounds, stagedir)
    @assert(rcf in overlapping_rcfs, "$rcf doesn't overlap with itself?")

    tic()
    println(rcf_bounds)
    results = one_node_infer(overlapping_rcfs,
                             stagedir;
                             objid=objid,  # just for making unit tests run fast
                             box=rcf_bounds,  # could exclude this argument
                             target_rcfs=[rcf,])
    Log.info("Inferred $rcf in $(toq()) seconds")

    fname = @sprintf "%s/celeste-%06d-%d-%04d.jld" outdir rcf.run rcf.camcol rcf.field
    JLD.save(fname, "results", results)
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
    elseif dt_nnodes > 1
        Log.debug("sky division parallelism")
        divide_sky_and_infer(box, stagedir; timing=times, outdir=outdir)
    else
        Log.debug("multithreaded parallelism only")
        tic()
        # Get vector of (run, camcol, field) triplets overlapping this patch
        rcfs = get_overlapping_fields(box, stagedir)
        times.query_fids = toq()

        results, obj_value = one_node_infer(rcfs, stagedir; box=box, timing=times)

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


"""
Estimates the amount of computation required to call `infer_box` on a
particular region of the sky.
"""
function estimate_box_runtime(box::BoundingBox, stagedir::String)
    rcfs = get_overlapping_fields(box, stagedir)
    catalog, targets, n_map, images = infer_init(rcfs, stagedir; box=box)

    # Typically we call `get_sky_patches()` with just a subset of the
    # catalog---just the light sources around one we're optimizing.
    # Here we call it for the whole catalog to get a count of the active
    # pixels all at once.
    patches = Infer.get_sky_patches(images, catalog)
    Infer.load_active_pixels!(images, patches)

    num_active = 0

    for n in 1:length(images), s in targets
        num_active += sum(patches[s, n].active_pixel_bitmap)
    end

    num_active
end

end
