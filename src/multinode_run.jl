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
# container for multinode-specific information
# ----------------
type MultinodeInfo
    dt::Dtree
    ni::Int
    ci::Int
    li::Int
    rundt::Bool
    lock::SpinLock
    nworkers::Int
    bar::CentSRBarrier
end

# ----------------
# container for bounding boxes and associated data
# ----------------
type BoxInfo
    state::Atomic{Int}
    box_idx::Int
    nsources::Int
    catalog::Vector{CatalogEntry}
    target_sources::Vector{Int}
    neighbor_map::Vector{Vector{Int}}
    images::Vector{Image}
    lock::SpinLock
    curr_source::Int
    sources_assignment::Vector{Vector{Vector{Int64}}}
    ea_vec::Vector{ElboArgs}
    vp_vec::Vector{VariationalParams{Float64}}
    cfg_vec::Vector{Config{DEFAULT_CHUNK,Float64}}
    ts_vp::Dict{Int64,Array{Float64}}

    BoxInfo() = new(Atomic{Int}(BoxDone), 0, 0, [], [], [], [], SpinLock(), 1, [],
                    [], [], [], Dict{Int64,Array{Float64}}())
end

# box states
const BoxDone = 0::Int
const BoxLoaded = 1::Int
const BoxInitializing = 2::Int
const BoxReady = 3::Int

# enable loading the next bounding box to process, while light sources
# in the current box are still being processed
const NumConcurrentBoxes = 2::Int


"""
Set up thread affinities for all ranks and create the Dtree scheduler
for distributing boxes to ranks.
"""
function init_multinode(nwi::Int)
    ranks_per_node = 1
    rpn = "?"
    if haskey(ENV, "CELESTE_RANKS_PER_NODE")
        rpn = ENV["CELESTE_RANKS_PER_NODE"]
        ranks_per_node = parse(Int, rpn)
        use_threads_per_core = 1
        if haskey(ENV, "CELESTE_THREADS_PER_CORE")
            use_threads_per_core = parse(Int, ENV["CELESTE_THREADS_PER_CORE"])
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
    end

    if grank() == 1
        Log.message("running with $rpn ranks per node, $(ngranks()) total ranks")
        Log.message("$nwi bounding boxes, ~$(ceil(Int, nwi / ngranks())) per rank")
    end

    dt, _ = Dtree(nwi, 0.4)
    ni, (ci, li) = initwork(dt)
    rundt = runtree(dt)
    lock = SpinLock()

    Log.debug("dtree: initial: $(ni) ($(ci) to $(li))")

    nworkers = nthreads() - (rundt ? 1 : 0)
    bar = setup_barrier(nworkers)

    return MultinodeInfo(dt, ni, ci, li, rundt, lock, nworkers, bar)
end


"""
Set up a global array to hold the results for each light source. Perform an
exclusive scan of the source counts array, to help locate the entry for each
source.
"""
function setup_results(boxes::Vector{BoundingBox}, box_source_counts::Vector{Int64})
    num_boxes = length(boxes)
    source_counts = zeros(Int, num_boxes+1)
    for i = 1:num_boxes
        box = boxes[i]
        source_counts[i+1] = source_counts[i] + box_source_counts[i]
    end
    num_sources = source_counts[num_boxes+1]

    if grank() == 1
        Log.message("$(num_sources) total sources in all boxes")
    end

    results = Garray(OptimizedSource, OptimizedSourceLen, num_sources)

    return results, source_counts
end


"""
Each node saves its portion of the results global array.
"""
function save_results(results::Garray, outdir::String)
    lo, hi = distribution(results, grank())
    lresults = access(results, lo, hi)
    fname = @sprintf("%s/celeste-multinode-rank%d.jld", outdir, grank())
    JLD.save(fname, "results", lresults)
    Log.message("saved results to $fname at $(Dates.now())")
end


"""
Determine the next bounding box to process and use `infer_init()` to
load its catalog, target sources, neighbor map, and images. Possibly
ask the scheduler for more box(es), if needed.
"""
function load_box(cbox::BoxInfo, mi::MultinodeInfo,
                  all_boxes::Vector{BoundingBox}, stagedir::String;
                  primary_initialization=true,
                  timing=InferTiming())
    tid = threadid()
    lock(cbox.lock)

    # another thread might have loaded this box already
    if cbox.state[] != BoxDone
        unlock(cbox.lock)
        return true
    end

    # determine which box to load next
    box_idx = 0
    while true
        lock(mi.lock)

        # the last item is 0 only when we're out of work
        if mi.li == 0
            unlock(mi.lock)
            unlock(cbox.lock)
            return false
        end

        # if we've run out of items, ask for more work
        if mi.ci > mi.li
            Log.debug("dtree: consumed allocation (last was $(mi.li))")
            mi.ni, (mi.ci, mi.li) = getwork(mi.dt)
            unlock(mi.lock)
            if mi.li == 0
                Log.debug("dtree: out of work")
            else
                Log.debug("dtree: $(mi.ni) work items ($(mi.ci) to $(mi.li))")
            end

        # otherwise, get the next box from our current allocation
        else
            box_idx = mi.ci
            mi.ci = mi.ci + 1
            unlock(mi.lock)
            break
        end
    end

    # load box `box_idx`
    @assert box_idx > 0
    box = all_boxes[box_idx]
    cbox.box_idx = box_idx


    # load the RCFs
    tic()
    rcfs = get_overlapping_fields(box, stagedir)
    rcftime = toq()
    timing.query_fids += rcftime

    # load catalog, target sources, neighbor map and images for these RCFs
    tic()
    cbox.catalog, cbox.target_sources, cbox.neighbor_map, cbox.images =
            infer_init(rcfs, stagedir;
                       box=box,
                       primary_initialization=primary_initialization,
                       timing=timing)
    loadtime = toq()

    cbox.nsources = length(cbox.target_sources)
    cbox.curr_source = 1
    cbox.state[] = BoxLoaded
    unlock(cbox.lock)

    Log.message("loaded box $(box.ramin), $(box.ramax), $(box.decmin), ",
                "$(box.decmax) ($(cbox.nsources) sources) in ",
                "$(rcftime + loadtime) secs")

    return true
end


"""
Thread function for running single inference on the light sources in
the specified bounding boxes. Used by the driver function.
"""
function single_infer_boxes(config::Configs.Config,
                            all_boxes::Vector{BoundingBox},
                            stagedir::String,
                            mi::MultinodeInfo,
                            all_results::Garray,
                            source_counts::Vector{Int},
                            conc_boxes::Vector{BoxInfo},
                            all_threads_timing::Vector{InferTiming},
                            primary_initialization=true)
    tid = threadid()
    all_threads_timing[tid] = InferTiming()
    thread_timing = all_threads_timing[tid]

    # Dtree parent ranks reserve one thread to drive the tree
    if mi.rundt && tid == nthreads()
        Log.debug("dtree: running tree")
        while runtree(mi.dt)
            Gasp.cpu_pause()
        end
        return
    end

    # all other threads process sources
    ts = 0
    ts_boxidx = 0
    curr_cbox = 1
    cbox = conc_boxes[curr_cbox]
    load_box(cbox, mi, all_boxes, stagedir;
             primary_initialization=primary_initialization,
             timing=thread_timing)
    while true
        # get the next source to process from the current box
        lock(cbox.lock)
        ts = cbox.curr_source
        ts_boxidx = cbox.box_idx
        cbox.curr_source = cbox.curr_source + 1
        unlock(cbox.lock)

        # if the current box is done, switch to the next box
        if ts > cbox.nsources
            # mark this box done
            if cbox.state[] != BoxDone
                atomic_cas!(cbox.state, BoxLoaded, BoxDone)
            end

            # switch to the next box (other threads may still be working here)
            curr_cbox = curr_cbox + 1
            if curr_cbox > NumConcurrentBoxes
                curr_cbox = 1
            end
            cbox = conc_boxes[curr_cbox]

            # if it hasn't already been loaded, load the next box to process
            if cbox.state[] != BoxLoaded
                if !load_box(cbox, mi, all_boxes, stagedir;
                            primary_initialization=primary_initialization,
                            timing=thread_timing)
                    break
                end
            end
            continue
        end

        # process the source and record the result
        try
            tic()
            result = process_source(config, ts, cbox.target_sources,
                                    cbox.catalog, cbox.neighbor_map,
                                    cbox.images)
            thread_timing.opt_srcs += toq()
            thread_timing.num_srcs += 1

            results_idx = source_counts[ts_boxidx] + ts
            tic()
            put!(all_results, results_idx, results_idx, [result])
            thread_timing.ga_put += toq()
        catch ex
            if is_production_run || nthreads() > 1
                Log.exception(ex)
            else
                rethrow(ex)
            end
        end

        # if we're close to the end of the sources in this box, start
        # loading the next box, to overlap loading with processing as
        # much as possible
        if mi.nworkers > 1 &&
                (ts == cbox.nsources - mi.nworkers ||
                (cbox.nsources <= mi.nworkers && ts == 1))
            curr_cbox = curr_cbox + 1
            if curr_cbox > NumConcurrentBoxes
                curr_cbox = 1
            end
            cbox = conc_boxes[curr_cbox]
            if cbox.state[] != BoxLoaded
                if !load_box(cbox, mi, all_boxes, stagedir;
                             primary_initialization=primary_initialization,
                             timing=thread_timing)
                    break
                end
            end
        end
    end
end


"""
Joint inference requires some initialization of a box: the sources must
be partitioned/batched, and persistent configurations allocated. All the
threads call this function, and may participate in initialization.
"""
function init_box(config::Configs.Config, cbox::BoxInfo, nworkers::Int;
                  cyclades_partition=true,
                  batch_size=400,
                  timing=InferTiming())
    # if this thread was late, move on quickly
    if cbox.state[] == BoxReady
        return
    end

    # only one thread does the partitioning and pre-allocation
    lock(cbox.lock)
    if cbox.state[] == BoxLoaded
        cbox.sources_assignment = partition_box(nworkers, cbox.target_sources,
                                        cbox.neighbor_map;
                                        cyclades_partition=cyclades_partition,
                                        batch_size=batch_size)
        cbox.ea_vec, cbox.vp_vec, cbox.cfg_vec, cbox.ts_vp =
                setup_vecs(cbox.nsources, cbox.target_sources, cbox.catalog)

        # update box state
        cbox.state[] = BoxInitializing
    end
    unlock(cbox.lock)

    # initialize elbo args for all sources
    tic()
    while cbox.state[] == BoxInitializing
        lock(cbox.lock)
        ts = cbox.curr_source
        cbox.curr_source = cbox.curr_source + 1
        unlock(cbox.lock)
        if ts > cbox.nsources
            if cbox.state[] == BoxInitializing
                atomic_cas!(cbox.state, BoxInitializing, BoxReady)
            end
            break
        end
        init_elboargs(config, ts, cbox.catalog, cbox.target_sources,
                      cbox.neighbor_map, cbox.images, cbox.ea_vec,
                      cbox.vp_vec, cbox.cfg_vec, cbox.ts_vp)
    end
    timing.init_elbo += toq()
end


"""
Thread function for running joint inference on the light sources in
the specified bounding boxes. Used by the driver function.
"""
function joint_infer_boxes(config::Configs.Config,
                           all_boxes::Vector{BoundingBox},
                           stagedir::String,
                           mi::MultinodeInfo,
                           all_results::Garray,
                           source_counts::Vector{Int},
                           conc_boxes::Vector{BoxInfo},
                           all_threads_timing::Vector{InferTiming},
                           primary_initialization::Bool,
                           cyclades_partition::Bool,
                           batch_size::Int,
                           within_batch_shuffling::Bool,
                           niters::Int)
    tid = threadid()
    all_threads_timing[tid] = InferTiming()
    thread_timing = all_threads_timing[tid]
    rng = MersenneTwister(42)

    # Dtree parent ranks reserve one thread to drive the tree
    if mi.rundt && tid == nthreads()
        Log.debug("dtree: running tree")
        while runtree(mi.dt)
            Gasp.cpu_pause()
        end
        return
    end

    # all other threads are workers
    curr_cbox = 0
    while true
        # load the next box
        curr_cbox = curr_cbox + 1
        if curr_cbox > NumConcurrentBoxes
            curr_cbox = 1
        end
        cbox = conc_boxes[curr_cbox]
        if cbox.state[] != BoxDone
            Log.warn("box state should be done, but is $(cbox.state[])?!")
        end
        if !load_box(cbox, mi, all_boxes, stagedir;
                     primary_initialization=primary_initialization,
                     timing=thread_timing)
            break
        end
        init_box(config, cbox, mi.nworkers;
                 cyclades_partition=cyclades_partition,
                 batch_size=batch_size,
                 timing=thread_timing)
        thread_barrier(mi.bar)

        # process sources in the box
        tic()
        nbatches = length(cbox.sources_assignment[1])
        try
            for iter = 1:niters
                for batch = 1:nbatches
                    # Shuffle the source assignments within each batch. This is
                    # disabled by default because it ruins the deterministic outcome
                    # required by the test cases.
                    # TODO: it isn't actually disabled by default?
                    if within_batch_shuffling
                        shuffle!(rng, cbox.sources_assignment[tid][batch])
                    end

                    for s in cbox.sources_assignment[tid][batch]
                        maximize!(cbox.ea_vec[s], cbox.vp_vec[s], cbox.cfg_vec[s])
                    end

                    # don't barrier on the last iteration
                    if !(iter == niters && batch == nbatches)
                        tic()
                        thread_barrier(mi.bar)
                        thread_timing.load_imba += toq()
                    end
                end
            end
        catch exc
            if is_production_run || nthreads() > 1
                Log.exception(exc)
            else
                rethrow()
            end
        end
        boxtime = toq()
        thread_timing.opt_srcs += boxtime
        if atomic_cas!(cbox.state, BoxReady, BoxDone) == BoxReady
            box = all_boxes[cbox.box_idx]
            Log.message("processed $(cbox.nsources) sources from box ",
                        "$(box.ramin), $(box.ramax), $(box.decmin), ",
                        "$(box.decmax) in $boxtime secs")
        end

        # each thread writes results for its sources
        nsrcs = 0
        tic()
        try
            for batch = 1:nbatches
                nsrcs = nsrcs + length(cbox.sources_assignment[tid][batch])
                for s in cbox.sources_assignment[tid][batch]
                    entry = cbox.catalog[cbox.target_sources[s]]
                    result = OptimizedSource(entry.thing_id,
                                             entry.objid,
                                             entry.pos[1],
                                             entry.pos[2],
                                             cbox.vp_vec[s][1])
                    results_idx = source_counts[cbox.box_idx] + s
                    put!(all_results, results_idx, results_idx, [result])
                end
            end
        catch exc
            if is_production_run || nthreads() > 1
                Log.exception(exc)
            else
                rethrow()
            end
        end
        thread_timing.ga_put += toq()
        thread_timing.num_srcs += nsrcs
    end
end


"""
Use Dtree to distribute the passed bounding boxes to multiple ranks for
processing. Within each rank, process the light sources in each of the
assigned boxes with multiple threads. This function drives both single
and joint inference.
"""
function multi_node_infer(all_boxes::Vector{BoundingBox},
                          box_source_counts::Vector{Int64},
                          stagedir::String;
                          outdir=".",
                          primary_initialization=true,
                          cyclades_partition=true,
                          batch_size=400,
                          within_batch_shuffling=true,
                          niters=3,
                          timing=InferTiming())
    Log.message("started at $(Dates.now())")

    # initialize scheduler, set up results global array
    nwi = length(all_boxes)
    mi = init_multinode(nwi)
    all_results, source_counts = setup_results(all_boxes, box_source_counts)

    # for concurrent box processing
    conc_boxes = [BoxInfo() for i=1:NumConcurrentBoxes]

    # number of threads that process sources
    nworkers = nthreads() - (mi.rundt ? 1 : 0)

    # per-thread timing
    all_threads_timing = Array{InferTiming}(nthreads())

    # inference configuration
    config = Configs.Config()

    # run the thread function
    if nthreads() == 1
        #single_infer_boxes(config, all_boxes, stagedir, mi, all_results,
        #                   source_counts, conc_boxes, all_threads_timing,
        #                   primary_initialization)
        joint_infer_boxes(config, all_boxes, stagedir, mi, all_results,
                          source_counts, conc_boxes, all_threads_timing,
                          primary_initialization, cyclades_partition,
                          batch_size, within_batch_shuffling, niters)
    else
        #ccall(:jl_threading_run, Void, (Any,),
        #      Core.svec(single_infer_boxes, config, all_boxes, stagedir, mi,
        #                all_results, source_counts, conc_boxes,
        #                all_threads_timing, primary_initialization))
        ccall(:jl_threading_run, Void, (Any,),
              Core.svec(joint_infer_boxes, config, all_boxes, stagedir, mi,
                        all_results, source_counts, conc_boxes,
                        all_threads_timing, primary_initialization,
                        cyclades_partition, batch_size, within_batch_shuffling,
                        niters))
    end
    Log.message("completed at $(Dates.now())")

    show_pixels_processed()

    # reduce and normalize collected timing information
    for i = 1:nthreads()
        add_timing!(timing, all_threads_timing[i])
    end
    timing.init_elbo /= mi.nworkers
    timing.opt_srcs /= mi.nworkers
    timing.load_imba /= mi.nworkers
    timing.ga_put /= mi.nworkers

    # output results
    tic()
    save_results(all_results, outdir)
    timing.write_results = toq()

    # shut down
    tic()
    finalize(mi.dt)
    timing.wait_done = toq()
    Log.message("synchronized ranks at $(Dates.now())")
end

