const TenXWork = false

# ----------------
# A simple centralized sense reversing thread barrier. Needed to allow the
# ordering constraints that Cyclades partitioning imposes on the processing
# of sources.
# ----------------

type CentSRBarrier
    thread_counter::Atomic{Int}
    num_threads::Int
    sense::Int
    thread_senses::Vector{Int}
end

function setup_barrier(num_threads::Int)
    CentSRBarrier(Atomic{Int}(num_threads), num_threads, 1, ones(Int, nthreads()))
end

function thread_barrier(bar::CentSRBarrier)
    tid = threadid()
    bar.thread_senses[tid] = 1 - bar.thread_senses[tid]
    sense = 1 - bar.sense
    if atomic_sub!(bar.thread_counter, 1) == 1
        bar.thread_counter[] = bar.num_threads
        bar.sense = 1 - bar.sense
    else
        while bar.thread_senses[tid] != bar.sense
            ccall(:jl_cpu_pause, Void, ())
            ccall(:jl_gc_safepoint, Void, ())
        end
    end
end


# ----------------
# container for multiprocessing/multithreading-specific information
# ----------------
type MultiInfo
    dt::Dtree
    ni::Int
    ci::Int
    li::Int
    rundt::Bool
    nworkers::Int
    bar::CentSRBarrier
end


# ----------------
# container for bounding boxes and associated data
# ----------------
type BoxInfo
    state::Atomic{Int}
    box_idx::Int
    catalog::Vector{CatalogEntry}
    target_sources::Vector{Int}
    neighbor_map::Vector{Vector{Int}}
    images::Vector{Image}
    source_rcfs::Vector{RunCamcolField}
    source_cat_idxs::Vector{Int16}
    sources_assignment::Vector{Vector{Vector{Int64}}}
    curr_cc::Atomic{Int}
    ea_vec::Vector{ElboArgs}
    vp_vec::Vector{VariationalParams{Float64}}
    cfg_vec::Vector{Config{DEFAULT_CHUNK,Float64}}
    ks::BoxKillSwitch

    BoxInfo() = new(Atomic{Int}(BoxDone), 0, [], [], [], [], [], [], [],
                    Atomic{Int}(1), [], [], [], BoxKillSwitch())
end

# box states
const BoxDone = 0::Int
const BoxReady = 1::Int
const BoxEnd = 2::Int


"""
Set thread affinities on all ranks, if so configured.
"""
function set_affinities()
    ranks_per_node = 1
    rpn = "?"
    if haskey(ENV, "CELESTE_RANKS_PER_NODE")
        rpn = ENV["CELESTE_RANKS_PER_NODE"]
        ranks_per_node = parse(Int, rpn)
        use_threads_per_core = 1
        if haskey(ENV, "CELESTE_THREADS_PER_CORE")
            use_threads_per_core = parse(Int, ENV["CELESTE_THREADS_PER_CORE"])
        else
            Log.one_message("WARN: assuming 1 thread per core, ",
                            "CELESTE_THREADS_PER_CORE not set")
        end
        if ccall(:jl_generating_output, Cint, ()) == 0
            lscpu = split(readstring(`lscpu`), '\n')
            cpus = parse(Int, split(lscpu[4], ':')[2])
            tpc = parse(Int, split(lscpu[6], ':')[2])
        else
            cpus = parse(Int, ENV["CELESTE_CPUS"])
            tpc = parse(Int, ENV["CELESTE_TPC"])
        end
        cores = div(cpus, tpc)
        affinitize(cores, tpc, ranks_per_node;
                   use_threads_per_core=use_threads_per_core)
    else
        Log.one_message("WARN: not affinitizing threads, ",
                        "CELESTE_RANKS_PER_NODE not set")
    end

    return rpn
end


"""
Create the Dtree scheduler for distributing boxes to ranks.
"""
function setup_multi(nwi::Int)
    dt, _ = Dtree(nwi, 0.25)
    ni, (ci, li) = initwork(dt)
    rundt = runtree(dt)

    Log.info("dtree: initial: $(ni) ($(ci) to $(li))")

    nworkers = nthreads() - (rundt ? 1 : 0) - 1
    bar = setup_barrier(max(1, nworkers))

    return MultiInfo(dt, ni, ci, li, rundt, nworkers, bar)
end


"""
Given an array of arrays of bounding boxes, determine how many primary sources
are in all the RCFs overlapped by those boxes. Use an exclusive scan of the
counts of sources in those RCFs (provided in `rcf_nsrcs`) to build a
dictionary mapping RCFs to their prefix sums. We use this dictionary to find
an RCF's sources' offset in a global array of all sources.
"""
function build_rcf_map(all_rcfs::Vector{RunCamcolField},
                       all_rcf_nsrcs::Vector{Int16},
                       all_boxes_rcf_idxs::Vector{Vector{Vector{Int32}}})
    # generate a list of all the indices of the RCFs that overlap all the boxes
    all_rcf_idxs = Vector{Int32}()
    for boxes_rcf_idxs in all_boxes_rcf_idxs
        for rcf_idxs in boxes_rcf_idxs
            append!(all_rcf_idxs, rcf_idxs)
        end
        all_rcf_idxs = unique(all_rcf_idxs)
    end

    # build the map for all these RCFs
    rcf_map = Dict{RunCamcolField,Int32}()
    num_rcfs = length(all_rcf_idxs)
    tot_srcs = 0
    for idx in all_rcf_idxs
        nsrcs = all_rcf_nsrcs[idx]
        rcf_map[all_rcfs[idx]] = tot_srcs
        tot_srcs += nsrcs
    end

    return tot_srcs, rcf_map
end


"""
Each node saves its portion of the results global array.
"""
function save_results(results::Garray, boxgroup::Int, outdir::String)
    try
        lo, hi = distribution(results, grank())
        lresults = access(results, lo, hi)
        fname = @sprintf("%s/celeste-multi-boxgroup%d-rank%d.jld", outdir,
                          boxgroup, grank())
        JLD.save(fname, "results", lresults)
        Log.message("$(Time(now())): saved results to $fname")
    catch exc
        Log.exception(exc)
    end
end


"""
Determine the next bounding box to process and load its catalog, target
sources, neighbor map, and images. Possibly ask the scheduler for more
box(es), if needed.
"""
function load_box(boxes::Vector{BoundingBox},
                  field_extents::Vector{FieldExtent},
                  strategy::SDSSIO.IOStrategy, mi::MultiInfo, cbox::BoxInfo,
                  timing::InferTiming)
    # expected: cbox.state[] == BoxDone

    # determine which box to load next
    box_idx = 0
    tic()
    while true
        # the last item is 0 only when we're out of work
        if mi.li == 0
            timing.sched_ovh += toq()
            return false
        end

        # if we've run out of items, ask for more work
        if mi.ci > mi.li
            mi.ni, (mi.ci, mi.li) = try getwork(mi.dt)
            catch exc
                Log.exception(exc)
                return false
            end
            if mi.li == 0
                Log.info("dtree: out of work")
            else
                Log.info("dtree: $(mi.ni) work items ($(mi.ci) to $(mi.li))")
            end

        # otherwise, get the next box from our current allocation
        else
            box_idx = mi.ci
            mi.ci += 1
            break
        end
    end
    timing.sched_ovh += toq()

    # load box `box_idx`
    box = boxes[box_idx]
    cbox.box_idx = box_idx

    # determine the RCFs
    rcfs = []
    try
        tic()
        rcfs = get_overlapping_fields(box, field_extents)
        rcftime = toq()
        timing.query_fids += rcftime
    catch exc
        Log.exception(exc)
    end

    # load the RCFs
    cbox.catalog = []
    cbox.target_sources = []
    cbox.neighbor_map = []
    cbox.images = []
    try
        tic()
        cbox.catalog, cbox.target_sources, cbox.neighbor_map,
            cbox.images, cbox.source_rcfs, cbox.source_cat_idxs =
                    infer_init(rcfs, strategy; box=box, timing=timing, )
        loadtime = toq()
    catch exc
        Log.exception(exc)
    end

    # set box information
    cbox.curr_cc[] = 1

    Log.message("$(Time(now())): loaded box #$(box_idx) ($(box.ramin), ",
                "$(box.ramax), $(box.decmin), $(box.decmax) ",
                "($(length(cbox.target_sources)) target sources)) in ",
                "$(rcftime + loadtime) secs")

    return true
end


"""
Joint inference requires some initialization of a box: the sources must
be partitioned and persistent configurations allocated. We also load
previous inference results if the global array is provided.
"""
function init_box(npartitions::Int, rcf_map::Dict{RunCamcolField,Int32},
                  prev_results, cbox::BoxInfo, timing::InferTiming)
    nsources = length(cbox.target_sources)
    cbox.ea_vec, cbox.vp_vec, cbox.cfg_vec, ts_vp =
            setup_vecs(nsources, cbox.target_sources, cbox.catalog)

    # to hold previously computed variational parameters (if any)
    cache = Dict{Int,Vector{Float64}}()

    # kill switch for this box
    cbox.ks = BoxKillSwitch()

    # initialize elbo args for all sources
    tic()
    for ts = 1:nsources
        entry_id = cbox.target_sources[ts]
        entry = cbox.catalog[entry_id]
        neighbor_ids = cbox.neighbor_map[ts]
        neighbors = cbox.catalog[neighbor_ids]

        cat_local = vcat([entry], neighbors)
        ids_local = vcat([entry_id], neighbor_ids)

        if TenXWork
            patches = Infer.get_sky_patches(cbox.images, cat_local, radius_override_pix=20.0)
        else
            patches = Infer.get_sky_patches(cbox.images, cat_local)
            Infer.load_active_pixels!(cbox.images, patches)
        end

        cbox.ea_vec[ts] = ElboArgs(cbox.images, patches, [1])

        vp = Vector{Float64}[haskey(ts_vp, x) ?
                             ts_vp[x] :
                             catalog_init_source(cbox.catalog[x])
                             for x in ids_local]
        cbox.cfg_vec[ts] = Config(cbox.ea_vec[ts], vp;
                                  termination_callback=cbox.ks)

        init_source(i) = i == 1 ?
                         generic_init_source(cat_local[i].pos) :
                         catalog_init_source(cat_local[i])

        if prev_results != nothing
            # we have previous results; load previously computed variational
            # parameters
            try
                cache_hits = 0
                ga_hits = 0
                loaded_vp = Vector{Vector{Float64}}(length(cat_local))
                for i = 1:length(cat_local)

                    # first check the cache
                    i_vp = get(cache, cat_local[i].thing_id, [])
                    if !isempty(i_vp)
                        cache_hits += 1
                        loaded_vp[i] = copy(i_vp)
                        continue
                    end

                    # it isn't in the cache; look in the global array
                    rcf = cbox.source_rcfs[ids_local[i]]
                    rofs = get(rcf_map, rcf, -1)
                    if rofs != -1
                        ri = cbox.source_cat_idxs[ids_local[i]]
                        ridx = convert(Int64, rofs + ri)
                        tic()
                        opt_src, osh = get(prev_results, ridx, ridx)
                        timing.ga_get += toq()
                        if isassigned(opt_src, 1)
                            ga_hits += 1
                            loaded_vp[i] = copy(opt_src[1].vs)
                            cache[opt_src[1].thingid] = copy(loaded_vp[i])
                            continue
                        else
                            #Log.debug("not found: $(cat_local[i].thing_id)")
                        end
                    else
                        Log.warn("no offset for RCF $(rcf.run), $(rcf.camcol), ",
                                 "$(rcf.field) (source $(entry.objid)) in map?")
                    end

                    # it isn't in the global array
                    loaded_vp[i] = vp[i]
                end
                vp = loaded_vp
                #Log.debug("got previous results for $ga_hits/$(length(cat_local)) ",
                #          "sources ($cache_hits cache hits)")
            catch exc
                Log.exception(exc)
            end
        end
        cbox.vp_vec[ts] = vp
    end
    timing.init_elbo += toq()
    cbox.sources_assignment = partition_box(npartitions, cbox.target_sources,
                                            cbox.neighbor_map, cbox.ea_vec)
    cbox.state[] = BoxReady
end


"""
Put inference results for a box into the global array.
"""
function store_box(cbox::BoxInfo, rcf_map::Dict{RunCamcolField,Int32},
                   results::Garray, timing::InferTiming)
    if cbox.ks.killed
        Log.message("$(Time(now())): abandoned box #$(cbox.box_idx)")
        return
    end

    tic()
    try
        for ts = 1:length(cbox.target_sources)
            entry = cbox.catalog[cbox.target_sources[ts]]
            result = OptimizedSource(entry.thing_id,
                                     entry.objid,
                                     entry.pos[1],
                                     entry.pos[2],
                                     cbox.vp_vec[ts][1])

            ci = cbox.target_sources[ts]
            rcf = cbox.source_rcfs[ci]
            rofs = get(rcf_map, rcf, -1)
            if rofs == -1
                Log.warn("no offset for RCF $(rcf.run), $(rcf.camcol), ",
                         "$(rcf.field) (source $(entry.objid)) in map?")
            else
                ri = cbox.source_cat_idxs[ci]
                ridx = convert(Int64, rofs + ri)
                #Log.debug("$(result.thingid) -> GA[$ridx]")
                tic()
                put!(results, ridx, ridx, [result])
                timing.ga_put += toq()
            end
        end
    catch exc
        if is_production_run || nthreads() > 1
            Log.exception(exc)
        else
            rethrow()
        end
    end
    timing.store_res += toq()
    timing.num_srcs += length(cbox.target_sources)

    if length(cbox.target_sources) > 0
        Log.message("$(Time(now())): completed box #$(cbox.box_idx)")
    end
end


"""
To be called by a thread dedicated to preloading boxes into `conc_boxes`.
"""
function preload_boxes(config::Configs.Config,
                       boxes::Vector{BoundingBox},
                       rcf_map::Dict{RunCamcolField,Int32},
                       field_extents::Vector{FieldExtent},
                       strategy::SDSSIO.IOStrategy,
                       mi::MultiInfo,
                       results::Garray,
                       prev_results,
                       conc_boxes::Vector{BoxInfo},
                       timing::InferTiming)
    curr_cbox = 1
    cbox = conc_boxes[curr_cbox]
    state = BoxReady

    while cbox.state[] != BoxEnd
        tic()
        while cbox.state[] != BoxDone
            Gasp.cpu_pause()
            ccall(:jl_gc_safepoint, Void, ())
        end
        timing.proc_wait += toq()

        # store box results
        try store_box(cbox, rcf_map, results, timing)
        catch exc
            Log.exception(exc)
        end

        # load and initialize box
        if state != BoxEnd
            if load_box(boxes, field_extents, strategy, mi, cbox, timing)
                try init_box(mi.nworkers, rcf_map, prev_results, cbox, timing)
                catch exc
                    Log.exception(exc)
                    if cbox.state[] != BoxReady
                        empty!(cbox.sources_assignment)
                        cbox.state[] = BoxReady
                    end
                end
            else
                state = BoxEnd
                cbox.state[] = BoxEnd
            end
        end

        # switch to the next box
        curr_cbox += 1
        if curr_cbox > length(conc_boxes)
            curr_cbox = 1
        end
        cbox = conc_boxes[curr_cbox]
    end
end


"""
Thread function for running joint inference on the light sources in
the specified bounding boxes.
"""
function joint_infer_boxes(config::Configs.Config,
                           boxes::Vector{BoundingBox},
                           rcf_map::Dict{RunCamcolField,Int32},
                           field_extents::Vector{FieldExtent},
                           strategy::SDSSIO.IOStrategy,
                           mi::MultiInfo,
                           results::Garray,
                           prev_results,
                           conc_boxes::Vector{BoxInfo},
                           all_threads_timing::Vector{InferTiming};
                           batch_size=7000,
                           within_batch_shuffling=true,
                           niters=3)
    tid = threadid()
    timing = all_threads_timing[tid]
    rng = MersenneTwister()

    # Dtree parent ranks reserve one thread to drive the tree
    if mi.rundt && tid == nthreads()
        Log.debug("$(Time(now())): dtree: running tree")
        while runtree(mi.dt)
            Gasp.cpu_pause()
        end
        return
    end

    # if we have at least one worker thread, we can use a preloader thread
    if mi.nworkers >= 1 && tid == 1
        preload_boxes(config, boxes, rcf_map, field_extents, strategy, mi,
                      results, prev_results, conc_boxes, timing)
        return
    end

    # all remaining threads are workers
    curr_cbox = 0
    while true
        # switch to the the next box
        curr_cbox = curr_cbox + 1
        if curr_cbox > length(conc_boxes)
            curr_cbox = 1
        end
        cbox = conc_boxes[curr_cbox]

        # prepare the box/wait for the box to be prepared
        if mi.nworkers == 0
            if !load_box(boxes, field_extents, strategy, mi, cbox, timing)
                break
            end
            init_box(mi.nworkers, rcf_map, prev_results, cbox, timing)
        else
            tic()
            while cbox.state[] == BoxDone
                Gasp.cpu_pause()
                ccall(:jl_gc_safepoint, Void, ())
            end
            timing.load_wait += toq()
            thread_barrier(mi.bar)
        end
        if cbox.state[] != BoxReady
            break
        end

        # process sources in the box
        tic()
        nbatches = length(cbox.sources_assignment)
        for iter = 1:niters
            for batch = 1:nbatches
                while true
                    cc = atomic_add!(cbox.curr_cc, 1)
                    if cc > length(cbox.sources_assignment[batch])
                        break
                    end

                    if within_batch_shuffling
                        shuffle!(rng, cbox.sources_assignment[batch][cc])
                    end

                    for ts in cbox.sources_assignment[batch][cc]
                        try maximize!(cbox.ea_vec[ts], cbox.vp_vec[ts],
                                      cbox.cfg_vec[ts])
                        catch exc
                            if is_production_run || nthreads() > 1
                                Log.exception(exc)
                            else
                                rethrow()
                            end
                        end
                        if cbox.ks.killed
                            break
                        end
                    end
                end
                cbox.ks.finished[tid] = time_ns()
                cbox.ks.lastfin[] = tid
                atomic_add!(cbox.ks.numfin, 1)

                tic()
                thread_barrier(mi.bar)
                timing.load_imba += toq()
                if cbox.ks.killed
                    break
                end
            end
            if cbox.ks.killed
                break
            end
        end
        boxtime = toq()
        timing.opt_srcs += boxtime

        cbox.state[] = BoxDone

        if mi.nworkers == 0
            store_box(cbox, rcf_map, results, timing)
        end
    end
end


"""
Use Dtree to distribute the passed bounding boxes to multiple ranks for
processing. Within each rank, process the light sources in each of the
assigned boxes with multiple threads. This function drives both single
and joint inference.
"""
function multi_node_infer(all_rcfs::Vector{RunCamcolField},
                          all_rcf_nsrcs::Vector{Int16},
                          all_boxes::Vector{Vector{BoundingBox}},
                          all_boxes_rcf_idxs::Vector{Vector{Vector{Int32}}},
                          strategy::SDSSIO.IOStrategy,
                          outdir::String)
    rpn = set_affinities()

    Log.one_message("$(Time(now())): Celeste started, $rpn ranks/node, ",
                    "$(ngranks()) total ranks, $(nthreads()) threads/rank")
    Log.one_message("  $(length(all_rcfs)) total RCFs\n",
                    "  $(length(all_boxes)) runs:")
    for i = 1:length(all_boxes)
        Log.one_message("    run $i: $(length(all_boxes[i])) boxes")
    end

    # determine number of concurrent boxes
    num_conc_boxes = 2
    if haskey(ENV, "CELESTE_NUM_CONC_BOXES")
        num_conc_boxes = parse(Int, ENV["CELESTE_NUM_CONC_BOXES"])
    end

    # load field extents
    field_extents = load_field_extents(strategy)

    # determine required size for the results global array and build the
    # offsets map into it to help locate a source
    tic()
    tot_srcs, rcf_map = build_rcf_map(all_rcfs, all_rcf_nsrcs, all_boxes_rcf_idxs)
    empty!(all_rcfs)
    empty!(all_rcf_nsrcs)
    empty!(all_boxes_rcf_idxs)
    gc()
    Log.one_message("  $(length(rcf_map)) RCFs overlap these boxes ",
                    "(map built in $(toq()) secs)\n",
                    "  $tot_srcs sources in these RCFs")

    # inference configuration
    config = Configs.Config()

    prev_results = nothing
    for i = 1:length(all_boxes)
        boxes = all_boxes[i]
        nwi = length(boxes)

        Log.one_message("$(Time(now())): starting run $i ($nwi boxes)")

        # timing per-run and per-thread
        timing = InferTiming()
        all_threads_timing = [InferTiming() for t = 1:nthreads()]

        # set up the scheduler and the global array for results
        mi = setup_multi(nwi)
        results = Garray(OptimizedSource, OptimizedSourceLen, tot_srcs)

        # for concurrent box processing
        conc_boxes = [BoxInfo() for i = 1:num_conc_boxes]

        # run the optimization
        if nthreads() == 1
            joint_infer_boxes(config, boxes, rcf_map,
                              field_extents, strategy, mi, results,
                              prev_results, conc_boxes, all_threads_timing)
        else
            ccall(:jl_threading_run, Void, (Any,),
                  Core.svec(joint_infer_boxes, config, boxes, rcf_map,
                            field_extents, strategy, mi, results,
                            prev_results, conc_boxes, all_threads_timing))
        end

        # write intermediate results to disk
        tic()
        save_results(results, i, outdir)
        timing.write_results = toq()
        prev_results = results

        show_pixels_processed()

        # shut down the scheduler
        tic()
        finalize(mi.dt)
        timing.wait_done = toq()
        Log.one_message("$(Time(now())): completed run $i")

        # reduce, normalize, and display collected timing information
        for i = 1:nthreads()
            add_timing!(timing, all_threads_timing[i])
        end
        nprocthreads = max(1, mi.nworkers)
        timing.load_wait /= nprocthreads
        timing.init_elbo /= nprocthreads
        timing.opt_srcs /= nprocthreads
        timing.load_imba /= nprocthreads
        timing.ga_get /= nprocthreads
        timing.ga_put /= nprocthreads
        puts_timing(timing)
    end
end

