module ParallelRun

using Base.Threads
using Base.Dates

import FITSIO
import JLD

import ..Configs
import ..Log
using ..Model
import ..SDSSIO
import ..Infer
import ..SDSSIO: RunCamcolField, IOStrategy, PlainFITSStrategy
import ..PSF

import ..DeterministicVI: infer_source

include("joint_infer.jl")

# In production mode, rather the development mode, always catch exceptions
const is_production_run = haskey(ENV, "CELESTE_PROD") && ENV["CELESTE_PROD"] != ""

# Use distributed parallelism (with Dtree)
const distributed = haskey(ENV, "USE_DTREE") && ENV["USE_DTREE"] != ""

if distributed
using Gasp
else
grank() = 1
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
# to time parts of Celeste
type InferTiming
    query_fids::Float64
    read_photoobj::Float64
    read_img::Float64
    preload_rcfs::Float64
    find_neigh::Float64
    load_wait::Float64
    proc_wait::Float64
    init_elbo::Float64
    opt_srcs::Float64
    num_srcs::Int64
    sched_ovh::Float64
    load_imba::Float64
    ga_get::Float64
    ga_put::Float64
    store_res::Float64
    write_results::Float64
    wait_done::Float64

    InferTiming() = new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0)
end

function add_timing!(i::InferTiming, j::InferTiming)
    i.query_fids = i.query_fids + j.query_fids
    i.read_photoobj = i.read_photoobj + j.read_photoobj
    i.read_img = i.read_img + j.read_img
    i.preload_rcfs = i.preload_rcfs + j.preload_rcfs
    i.find_neigh = i.find_neigh + j.find_neigh
    i.load_wait = i.load_wait + j.load_wait
    i.proc_wait = i.proc_wait + j.proc_wait
    i.init_elbo = i.init_elbo + j.init_elbo
    i.opt_srcs = i.opt_srcs + j.opt_srcs
    i.num_srcs = i.num_srcs + j.num_srcs
    i.sched_ovh = i.sched_ovh + j.sched_ovh
    i.load_imba = i.load_imba + j.load_imba
    i.ga_get = i.ga_get + j.ga_get
    i.ga_put = i.ga_put + j.ga_put
    i.store_res = i.store_res + j.store_res
    i.write_results = i.write_results + j.write_results
    i.wait_done = i.wait_done + j.wait_done
end

function puts_timing(i::InferTiming)
    i.num_srcs = max(1, i.num_srcs)
    Log.message("timing: query_fids=$(i.query_fids)")
    Log.message("timing: read_photoobj=$(i.read_photoobj)")
    Log.message("timing: read_img=$(i.read_img)")
    Log.message("timing: preload_rcfs=$(i.preload_rcfs)")
    Log.message("timing: find_neigh=$(i.find_neigh)")
    Log.message("timing: load_wait=$(i.load_wait)")
    #Log.message("timing: proc_wait=$(i.proc_wait)")
    Log.message("timing: init_elbo=$(i.init_elbo)")
    Log.message("timing: opt_srcs=$(i.opt_srcs)")
    Log.message("timing: num_srcs=$(i.num_srcs)")
    #Log.message("timing: average opt_srcs=$(i.opt_srcs/i.num_srcs)")
    #Log.message("timing: sched_ovh=$(i.sched_ovh)")
    Log.message("timing: load_imba=$(i.load_imba)")
    #Log.message("timing: ga_get=$(i.ga_get)")
    #Log.message("timing: ga_put=$(i.ga_put)")
    #Log.message("timing: store_res=$(i.store_res)")
    Log.message("timing: write_results=$(i.write_results)")
    Log.message("timing: wait_done=$(i.wait_done)")
end

function time_puts(elapsedtime, bytes, gctime, allocs)
    s = @sprintf("timing: total=%10.6f seconds", elapsedtime/1e9)
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
    Log.message(s)
end


# ------
# initialization helpers

"""
Given a list of RCFs, load the catalogs, determine the target sources,
load the images, and build the neighbor map.
"""
function infer_init(rcfs::Vector{RunCamcolField},
                    strategy::SDSSIO.IOStrategy;
                    objid="",
                    box=BoundingBox(-1000., 1000., -1000., 1000.),
                    primary_initialization=true,
                    timing=InferTiming())
    catalog = Vector{CatalogEntry}()
    target_sources = Vector{Int}()
    neighbor_map = Vector{Vector{Int}}()
    images = Vector{Image}()
    source_rcfs = Vector{RunCamcolField}()
    source_cat_idxs = Vector{Int16}()

    # Read all primary objects in these fields.
    duplicate_policy = primary_initialization ? :primary : :first
    tic()
    states = SDSSIO.preload_rcfs(strategy, rcfs)
    timing.preload_rcfs += toq()

    tic()
    for (i,(rcf,state)) in enumerate(zip(rcfs, states))
        this_cat = SDSSIO.read_photoobj(strategy, rcf, state, duplicate_policy=duplicate_policy)
        these_sources_rcfs = Vector{RunCamcolField}(length(this_cat))
        fill!(these_sources_rcfs, rcf)
        these_sources_idxs = collect(Int16, 1:length(this_cat))
        append!(catalog, this_cat)
        append!(source_rcfs, these_sources_rcfs)
        append!(source_cat_idxs, these_sources_idxs)
    end
    timing.read_photoobj += toq()

    # Get indices of entries in the RA/Dec range of interest.
    entry_in_range = entry->((box.ramin < entry.pos[1] < box.ramax) &&
                             (box.decmin < entry.pos[2] < box.decmax))
    target_sources = find(entry_in_range, catalog)

    Log.info("$(Time(now())): $(length(catalog)) primary sources, ",
             "$(length(target_sources)) target sources in $(box.ramin), ",
             "$(box.ramax), $(box.decmin), $(box.decmax)")

    # Filter any object not specified, if an objid is specified
    if objid != ""
        target_sources = filter(ts->(catalog[ts].objid == objid), target_sources)
        Log.info("$(length(target_sources)) target light sources after objid cut")
    end

    # Load images and neighbor map for target sources
    if length(target_sources) > 0
        # Read in images for all (run, camcol, field).
        try
            tic()
            images = SDSSIO.load_field_images(strategy, rcfs, states)
            timing.read_img += toq()
        catch ex
            Log.exception(ex)
            empty!(target_sources)
        end

        tic()
        neighbor_map = Infer.find_neighbors(target_sources, catalog, images)
        timing.find_neigh += toq()
    end

    return catalog, target_sources, neighbor_map, images,
           source_rcfs, source_cat_idxs
end


# ------
# optimization result container

immutable OptimizedSource
    thingid::Int64
    objid::String
    init_ra::Float64
    init_dec::Float64
    vs::Vector{Float64}
end
const OptimizedSourceLen = 465

function serialize(s::Base.AbstractSerializer, os::OptimizedSource)
    Base.serialize_type(s, typeof(os))
    write(s.io, os.thingid)
    @assert length(os.objid) == 19
    for i = 1:19
        write(s.io, os.objid.data[i])
    end
    write(s.io, os.init_ra)
    write(s.io, os.init_dec)
    for i = 1:length(Celeste.Model.ids)
        write(s.io, os.vs[i])
    end
end

function deserialize(s::Base.AbstractSerializer, ::Type{OptimizedSource})
    thingid = read(s.io, Int64)::Int64
    objid_data = zeros(UInt8, 19)
    for i = 1:19
        objid_data[i] = read(s.io, UInt8)::UInt8
    end
    init_ra = read(s.io, Float64)::Float64
    init_dec = read(s.io, Float64)::Float64
    vs = zeros(Float64, length(Celeste.Model.ids))
    for i = 1:length(Celeste.Model.ids)
        vs[i] = read(s.io, Float64)
    end
    OptimizedSource(thingid, String(objid_data), init_ra, init_dec, vs)
end


# ------
# optimization

"""
Optimize the `ts`th element of `target_sources`.
"""
function process_source(config::Configs.Config,
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
    Log.info("$(entry.objid): $(toq()) secs")
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
Use multiple threads on one node to fit the Celeste model to sources in a given
bounding box.
"""
function one_node_infer(rcfs::Vector{RunCamcolField},
                        stagedir::String;
                        infer_callback=one_node_single_infer,
                        objid="",
                        box=BoundingBox(-1000., 1000., -1000., 1000.),
                        primary_initialization=true,
                        timing=InferTiming())
    strategy = PlainFITSStrategy(stagedir)

    catalog, target_sources, neighbor_map, images =
        infer_init(rcfs,
                   strategy;
                   objid=objid,
                   box=box,
                   primary_initialization=primary_initialization)

    Log.info("running with $(nthreads()) threads")

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
get_overlapping_field_extents(query::BoundingBox, strategy::SDSSIO.IOStrategy) =
    get_overlapping_field_extents(query, strategy.stagedir)


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
type FieldExtent
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
function load_field_extents(stagedir::String)
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

    fes = Vector{FieldExtent}()
    for i = 1:length(all_run)
        fe = FieldExtent(all_run[i], all_camcol[i], all_field[i],
                         all_ramin[i], all_ramax[i],
                         all_decmin[i], all_decmax[i])
        push!(fes, fe)
    end

    return fes
end
load_field_extents(strategy::SDSSIO.IOStrategy) = load_field_extents(strategy.stagedir)


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
    fname = @sprintf("%s/celeste-%.4f-%.4f-%.4f-%.4f-rank%d.jld",
                     outdir, ramin, ramax, decmin, decmax, grank())
    JLD.save(fname, "results", results)
end

save_results(outdir, box, results) =
    save_results(outdir, box.ramin, box.ramax, box.decmin, box.decmax, results)


"""
called from main entry point.
"""
function infer_box(box::BoundingBox, stagedir::String, outdir::String)
    strategy = PlainFITSStrategy(stagedir)

    timing = InferTiming()

    Log.info("processing box $(box.ramin), $(box.ramax), $(box.decmin), ",
             "$(box.decmax) with $(nthreads()) threads")

    @time begin
        tic()
        # Get vector of (run, camcol, field) triplets overlapping this patch
        rcfs = get_overlapping_fields(box, strategy)
        timing.query_fids = toq()

        catalog, target_sources, neighbor_map, images =
            infer_init(rcfs,
                       strategy;
                       box=box,
                       primary_initialization=true,
                       timing=timing)

        #results = one_node_single_infer(catalog, target_sources,
        #                                neighbor_map, images; timing=timing)
        results = one_node_joint_infer(catalog, target_sources, neighbor_map,
                                       images; timing=timing)

        tic()
        save_results(outdir, box, results)
        timing.write_results = toq()
    end
    puts_timing(timing)
end


if distributed
include("multinode_run.jl")
else
function multi_node_infer(all_rcfs::Vector{RunCamcolField},
                          all_rcf_nsrcs::Vector{Int16},
                          all_boxes::Vector{Vector{BoundingBox}},
                          all_boxes_rcf_idxs::Vector{Vector{Vector{Int32}}},
                          strategy::SDSSIO.IOStrategy,
                          outdir::String)
    Log.one_message("ERROR: distributed functionality is disabled ",
                    "(set USE_DTREE=1 to enable)")
    exit(-1)
end
end


"""
called from main entry point.
"""
function infer_boxes(all_rcfs::Vector{RunCamcolField},
                     all_rcf_nsrcs::Vector{Int16},
                     all_boxes::Vector{Vector{BoundingBox}},
                     all_boxes_rcf_idxs::Vector{Vector{Vector{Int32}}},
                     strategy::SDSSIO.IOStrategy,
                     outdir::String)
    # Base.@time hack for distributed environment
    gc_stats = ()
    gc_diff_stats = ()
    gc_stats = Base.gc_num()
    elapsed_time = time_ns()

    multi_node_infer(all_rcfs, all_rcf_nsrcs, all_boxes, all_boxes_rcf_idxs,
                     strategy, outdir)

    # Base.@time hack for distributed environment
    elapsed_time = time_ns() - elapsed_time
    gc_diff_stats = Base.GC_Diff(Base.gc_num(), gc_stats)
    time_puts(elapsed_time, gc_diff_stats.allocd, gc_diff_stats.total_time,
              Base.gc_alloc_count(gc_diff_stats))
end

end

