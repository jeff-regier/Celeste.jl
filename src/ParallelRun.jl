module ParallelRun

using Base.Threads

import FITSIO
import JLD

import ..Configs
import ..Log
using ..Model
import ..SDSSIO
import ..Infer
import ..SDSSIO: RunCamcolField
import ..PSF

import ..DeterministicVI: infer_source

include("joint_infer.jl")

# In production mode, rather the development mode, always catch exceptions
const is_production_run = haskey(ENV, "CELESTE_PROD") && ENV["CELESTE_PROD"] != ""

# Use distributed parallelism (with Dtree)
const distributed = haskey(ENV, "USE_DTREE") && ENV["USE_DTREE"] != ""

if distributed
import Gasp.nodeid
else
nodeid = 1
end

#
# ------ bounding box ------
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


# ------
# thread-safe print functions
@inline nputs(nid, s...) = ccall(:puts, Cint, (Cstring,), string("[$nid] ", s...))
@inline ntputs(nid, tid, s...) = ccall(:puts, Cint, (Cstring,), string("[$nid]<$tid> ", s...))

@inline phalse(b) = b[] = false


# ------
# timed parts of Celeste
type InferTiming
    query_fids::Float64
    num_infers::Int64
    read_photoobj::Float64
    read_img::Float64
    opt_srcs::Float64
    num_srcs::Int64
    write_results::Float64
    wait_done::Float64

    InferTiming() = new(0.0, 0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)
end

function add_timing!(i::InferTiming, j::InferTiming)
    i.query_fids = i.query_fids + j.query_fids
    i.num_infers = i.num_infers + j.num_infers
    i.read_photoobj = i.read_photoobj + j.read_photoobj
    i.read_img = i.read_img + j.read_img
    i.opt_srcs = i.opt_srcs + j.opt_srcs
    i.num_srcs = i.num_srcs + j.num_srcs
    i.write_results = i.write_results + j.write_results
    i.wait_done = i.wait_done + j.wait_done
end

function puts_timing(i::InferTiming)
    i.num_srcs = max(1, i.num_srcs)
    nputs(nodeid, "timing: query_fids=$(i.query_fids)")
    nputs(nodeid, "timing: num_infers=$(i.num_infers)")
    nputs(nodeid, "timing: read_photoobj=$(i.read_photoobj)")
    nputs(nodeid, "timing: read_img=$(i.read_img)")
    nputs(nodeid, "timing: opt_srcs=$(i.opt_srcs)")
    nputs(nodeid, "timing: num_srcs=$(i.num_srcs)")
    nputs(nodeid, "timing: average opt_srcs=$(i.opt_srcs/i.num_srcs)")
    nputs(nodeid, "timing: write_results=$(i.write_results)")
    nputs(nodeid, "timing: wait_done=$(i.wait_done)")
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


function infer_init(rcfs::Vector{RunCamcolField},
                    stagedir::String;
                    objid="",
                    box=BoundingBox(-1000., 1000., -1000., 1000.),
                    primary_initialization=true,
                    timing=InferTiming())
    # Read all primary objects in these fields.
    duplicate_policy = primary_initialization ? :primary : :first
    tic()
    catalog = SDSSIO.read_photoobj_files(rcfs, stagedir,
                        duplicate_policy=duplicate_policy)
    timing.read_photoobj = toq()
    Log.info("$(length(catalog)) primary sources")

    # Get indices of entries in the RA/Dec range of interest.
    entry_in_range = entry->((box.ramin < entry.pos[1] < box.ramax) &&
                             (box.decmin < entry.pos[2] < box.decmax))
    target_sources = find(entry_in_range, catalog)

    Log.info(string("Found $(length(target_sources)) target sources in ",
          "$(box.ramin), $(box.ramax), $(box.decmin), $(box.decmax)"))

    # Filter any object not specified, if an objid is specified
    if objid != ""
        target_sources = filter(ts->(catalog[ts].objid == objid), target_sources)
        Log.info("$(length(target_sources)) target light sources after objid cut")
    end

    # Load images and neighbor map for target sources
    images = Image[]
    neighbor_map = Vector{Int}[]
    if length(target_sources) > 0
        # Read in images for all (run, camcol, field).
        try
            tic()
            images = SDSSIO.load_field_images(rcfs, stagedir)
            timing.read_img = toq()
        catch ex
            Log.error(string(ex))
        end

        tic()
        neighbor_map = Infer.find_neighbors(target_sources, catalog, images)
        Log.info("neighbors found in $(toq()) seconds")
    end

    return catalog, target_sources, neighbor_map, images
end


immutable OptimizedSource
    thingid::Int64
    objid::String
    init_ra::Float64
    init_dec::Float64
    vs::Vector{Float64}
end


"""
Optimize the `ts`th element of `sources`.
"""
function process_source(config::Configs.Config,
                        ts::Int,
                        target_sources::Vector{Int},
                        catalog::Vector{CatalogEntry},
                        neighbor_map::Vector{Vector{Int}},
                        images::Vector{Image};
                        infer_source_callback=infer_source)
    s = target_sources[ts]
    entry = catalog[s]

    neighbors = catalog[neighbor_map[ts]]

    tic()
    vs_opt = infer_source_callback(config, images, neighbors, entry)
    ntputs(nodeid, threadid(), "processed objid $(entry.objid) in $(toq()) secs")
    return OptimizedSource(entry.thing_id,
                           entry.objid,
                           entry.pos[1],
                           entry.pos[2],
                           vs_opt)
end


"""
Use multiple threads to process each target source with the specified
callback and write the results to a file.
"""
function one_node_single_infer(config::Configs.Config,
                               catalog::Vector{CatalogEntry},
                               target_sources::Vector{Int},
                               neighbor_map::Vector{Vector{Int}},
                               images::Vector{Image};
                               infer_source_callback=infer_source,
                               timing=InferTiming())
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
                result = process_source(config, ts, target_sources, catalog, neighbor_map,
                                        images;
                                        infer_source_callback=infer_source_callback)

                lock(results_lock)
                push!(results, result)
                unlock(results_lock)
            catch ex
                if is_production_run || nthreads() > 1
                    Log.error(string(ex))
                else
                    rethrow(ex)
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

    return results
end

# legacy wrapper
function one_node_single_infer(catalog::Vector{CatalogEntry},
                               target_sources::Vector{Int},
                               neighbor_map::Vector{Vector{Int}},
                               images::Vector{Image};
                               infer_source_callback=infer_source,
                               timing=InferTiming())
    one_node_single_infer(
        Configs.Config(),
        catalog,
        target_sources,
        neighbor_map,
        images,
        infer_source_callback=infer_source_callback,
        timing=timing,
    )
end

"""
Use mulitple threads on one node to fit the Celeste model to sources in a given
bounding box.
"""
function one_node_infer(rcfs::Vector{RunCamcolField},
                        stagedir::String;
                        infer_callback=one_node_single_infer,
                        objid="",
                        box=BoundingBox(-1000., 1000., -1000., 1000.),
                        primary_initialization=true,
                        timing=InferTiming())
    catalog, target_sources, neighbor_map, images =
        infer_init(rcfs,
                   stagedir;
                   objid=objid,
                   box=box,
                   primary_initialization=primary_initialization)

    Log.info("Running with $(nthreads()) threads")

    # NB: All I/O happens above. The methods below don't touch disk.
    infer_callback(catalog, target_sources, neighbor_map, images)
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
Save provided results to a JLD file.
"""
function save_results(outdir, ramin, ramax, decmin, decmax, results)
    fname = @sprintf("%s/celeste-%.4f-%.4f-%.4f-%.4f-node%d.jld",
                     outdir, ramin, ramax, decmin, decmax, nodeid)
    JLD.save(fname, "results", results)
end

save_results(outdir, box, results) =
    save_results(outdir, box.ramin, box.ramax, box.decmin, box.decmax, results)


"""
called from main entry point.
"""
function infer_box(box::BoundingBox, stagedir::String, outdir::String;
                   timing=InferTiming())
    Log.debug("multithreaded parallelism only")

    tic()
    # Get vector of (run, camcol, field) triplets overlapping this patch
    rcfs = get_overlapping_fields(box, stagedir)
    timing.query_fids = toq()

    results = one_node_infer(rcfs, stagedir; box=box, timing=timing)

    tic()
    save_results(outdir, box, results)
    timing.write_results = toq()

    puts_timing(timing)
end


if distributed
include("multinode_run.jl")
else
function multi_node_infer(boxes::Vector{BoundingBox},
                          stagedir::String;
                          outdir=".",
                          primary_initialization=true,
                          timing=InferTiming())
    Log.error("distributed functionality is disabled (set USE_DTREE=1 to enable)")
    exit(-1)
end
end


"""
called from main entry point.
"""
function infer_boxes(boxes::Vector{BoundingBox}, stagedir::String, outdir::String;
                     timing = InferTiming())
    # Base.@time hack for distributed environment
    gc_stats = ()
    gc_diff_stats = ()
    elapsed_time = 0.0
    gc_stats = Base.gc_num()
    elapsed_time = time_ns()

    multi_node_infer(boxes, stagedir; outdir=outdir, timing=timing)

    # Base.@time hack for distributed environment
    elapsed_time = time_ns() - elapsed_time
    gc_diff_stats = Base.GC_Diff(Base.gc_num(), gc_stats)
    time_puts(elapsed_time, gc_diff_stats.allocd, gc_diff_stats.total_time,
              Base.gc_alloc_count(gc_diff_stats))

    puts_timing(timing)
end

end
