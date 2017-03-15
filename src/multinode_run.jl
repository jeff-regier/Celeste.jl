# container for multinode-specific information
type MultinodeInfo
    dt::Dtree
    ni::Int
    ci::Int
    li::Int
    rundt::Bool
    lock::SpinLock
end

# one of these is used for every bounding box
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
    #ea_vec::Vector{ElboArgs}
    #vp_vec::Vector{VariationalParams{Float64}}
    #cfg_vec::Vector{Config{DEFAULT_CHUNK,Float64}}
    #ts_vp::Dict{Int64,Array{Float64}}

    BoxInfo() = new(Atomic{Int}(BoxDone), 0, 0, [], [], [], [], SpinLock(), 1, [])
                    #[], [], [], Dict{Int64,Array{Float64}}())
end

# box states
const BoxDone = 0::Int
const BoxLoaded = 1::Int
const BoxInitializing = 2::Int
const BoxReady = 3::Int

# enable loading the next bounding box to process, while light sources
# in the current box are still being processed
const NumConcurrentBoxes = 2::Int


function show_affinities()
    function show_affinity()
        tid = threadid()
        cpu = ccall(:sched_getcpu, Cint, ())
        ccall(:puts, Cint, (Cstring,), string("[$(grank())]<$tid> => $cpu"))
    end
    ccall(:jl_threading_run, Void, (Any,), Core.svec(show_affinity))
end


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

    return MultinodeInfo(dt, ni, ci, li, rundt, lock)
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

    Log.message("loading box $(box.ramin), $(box.ramax), $(box.decmin), $(box.decmax)")

    # load the RCFs
    tic()
    rcfs = get_overlapping_fields(box, stagedir)
    timing.query_fids = timing.query_fids + toq()

    # load catalog, target sources, neighbor map and images for these RCFs
    cbox.catalog, cbox.target_sources, cbox.neighbor_map, cbox.images =
            infer_init(rcfs, stagedir;
                       box=box,
                       primary_initialization=primary_initialization,
                       timing=timing)
    cbox.nsources = length(cbox.target_sources)
    cbox.curr_source = 1
    Log.message("$(cbox.nsources) sources in box")
    cbox.state[] = BoxLoaded

    unlock(cbox.lock)
    return true
end


"""
Run single-inference for the light sources in each of the specified
bounding boxes. A dynamic scheduler distributes boxes to ranks in
order to balance load.
"""
function multi_node_single_infer(all_boxes::Vector{BoundingBox},
                                 box_source_counts::Vector{Int64},
                                 stagedir::String;
                                 outdir=".",
                                 primary_initialization=true,
                                 timing=InferTiming())
    Log.message("started at $(Dates.now())")

    # initialize scheduler, set up results global array
    nwi = length(all_boxes)
    mi = init_multinode(nwi)
    all_results, source_counts = setup_results(all_boxes, box_source_counts)

    # for concurrent box processing
    conc_boxes = [BoxInfo() for i=1:NumConcurrentBoxes]

    nworkers = nthreads() - (mi.rundt ? 1 : 0)

    # per-thread timing
    thread_timing = Array{InferTiming}(nthreads())

    config = Configs.Config()

    # thread function to process sources in boxes
    function process_boxes()
        tid = threadid()
        thread_timing[tid] = InferTiming()
        timing = thread_timing[tid]

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
                 timing=timing)
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
                                timing=timing)
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
                timing.opt_srcs += toq()
                timing.num_srcs += 1

                results_idx = source_counts[ts_boxidx] + ts
                tic()
                put!(all_results, results_idx, results_idx, [result])
                timing.ga_put += toq()
                #Log.debug("wrote result to entry $(results_idx) of $(source_counts[end])")
            catch ex
                if is_production_run || nthreads() > 1
                    Log.exception(ex)
                else
                    rethrow(ex)
                end
            end

            # if we're close to the end of the sources in this box, start loading
            # the next box, to overlap loading with processing as much as possible
            if nworkers > 1 &&
                    (ts == cbox.nsources - nworkers ||
                    (cbox.nsources <= nworkers && ts == 1))
                curr_cbox = curr_cbox + 1
                if curr_cbox > NumConcurrentBoxes
                    curr_cbox = 1
                end
                cbox = conc_boxes[curr_cbox]
                if cbox.state[] != BoxLoaded
                    if !load_box(cbox, mi, all_boxes, stagedir;
                                 primary_initialization=primary_initialization,
                                 timing=timing)
                        break
                    end
                end
            end
        end
    end

    if nthreads() == 1
        process_boxes()
    else
        ccall(:jl_threading_run, Void, (Any,), Core.svec(process_boxes))
        #ccall(:jl_threading_profile, Void, ())
    end
    Log.message("completed at $(Dates.now())")

    for i = 1:nthreads()
        add_timing!(timing, thread_timing[i])
    end

    tic()
    save_results(all_results, outdir)
    timing.write_results = toq()

    tic()
    finalize(mi.dt)
    timing.wait_done = toq()
    Log.message("synchronized ranks at $(Dates.now())")
end


"""
Return a Cyclades partitioning or an equal partitioning of the target sources.
"""
function partition_box(npartitions::Int, target_sources::Vector{Int},
                       neighbor_map::Vector{Vector{Int}};
                       cyclades=true,
                       batch_size=400)
    if cyclades
        cyclades_neighbor_map = Dict{Int64,Vector{Int64}}()
        for (index, neighbors) in enumerate(neighbor_map)
            cyclades_neighbor_map[target_sources[index]] = neighbors
        end
        return partition_cyclades(npartitions, target_sources, cyclades_neighbor_map,
                                  batch_size=batch_size)
    else
        return partition_equally(npartitions, length(target_sources))
    end
end


"""
Initialize elbo args for the specified target source.
"""
function init_elboargs(ts::Int, catalog::Vector{CatalogEntry},
                       target_sources::Vector{Int},
                       neighbor_map::Vector{Vector{Int}}, images::Vector{Image},
                       ts_vp::Dict{Int64,Array{Float64}},
                       ea_vec::Vector{ElboArgs},
                       cfg_vec::Vector{Config{DEFAULT_CHUNK,Float64}};
                       use_fft=false,
                       trust_region=NewtonTrustRegion())
    entry_id = target_sources[ts]
    entry = catalog[entry_id]
    neighbor_ids = neighbor_map[ts]
    neighbors = catalog[neighbor_ids]
    cat_local = vcat([entry], neighbors)
    ids_local = vcat([entry_id], neighbor_ids)
    try
        patches = Infer.get_sky_patches(images, cat_local)
        Infer.load_active_pixels!(images, patches)

        # Load vp with shared target source params, and also vp
        # that doesn't share target source params
        vp = Vector{Float64}[haskey(ts_vp, x) ?
                             ts_vp[x] :
                             catalog_init_source(catalog[x])
                             for x in ids_local]

        # Switch parameters based on whether or not we're using the fft method
        if use_fft
            ea, _ = initialize_fft_elbo_parameters(images, vp, patches, [1],
                                                   use_raw_psf=false,
                                                   allocate_fsm_mat=true)
        else
            ea = ElboArgs(images, vp, patches, [1])
        end

        ea_vec[ts] = ea
        cfg_vec[ts] = Config(ea, trust_region=trust_region)
    catch ex
        if is_production_run || nthreads() > 1
            Log.exception(ex)
        else
            rethrow(ex)
        end
    end
end


"""
Joint inference requires some initialization of a box: the sources must
be partitioned/batched, and persistent configurations allocated. All the
threads call this function, and may participate in initialization.
"""
function init_box(cbox::BoxInfo, nworkers::Int;
                  cyclades=true,
                  batch_size=400,
                  use_fft=false,
                  trust_region=NewtonTrustRegion(),
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
                                                cyclades=cyclades,
                                                batch_size=batch_size)

        # configurations need to be persisted across calls to maximize! so
        # that location constraints don't shift from their initial position
        cbox.ea_vec = Vector{ElboArgs}(cbox.nsources)
        cbox.cfg_vec = Vector{Config{DEFAULT_CHUNK,Float64}}(cbox.nsources)

        # pre-allocate elbo args variational params
        cbox.ts_vp = Dict{Int64,Array{Float64}}()
        for ts in cbox.target_sources
            cat = cbox.catalog[ts]
            cbox.ts_vp[ts] = generic_init_source(cat.pos)
        end

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
        else
            #Log.debug("initializing source $ts")
            init_elboargs(ts, cbox.catalog, cbox.target_sources, cbox.neighbor_map,
                          cbox.images, cbox.ts_vp, cbox.ea_vec, cbox.cfg_vec;
                          use_fft=use_fft,
                          trust_region=trust_region)
        end
    end
    timing.init_elbo = timing.init_elbo + toq()
end


"""
Called by a thread to process a batch of sources in a box.
"""
function process_sources(sources::Vector{Int64}, cbox::BoxInfo;
                         use_fft=false,
                         within_batch_shuffling=true)
    for s in sources
        try
            ea, cfg = cbox.ea_vec[s], cbox.cfg_vec[s]
            if use_fft
                f = FFTElboFunction(load_fsm_mat(ea, cbox.images; use_raw_psf=true))
            else
                f = DeterministicVI.elbo
            end
            Log.debug("maximizing source $s")
            NewtonMaximize.maximize!(f, ea, cfg)
        catch ex
            if is_production_run || nthreads() > 1
                Log.exception(ex)
            else
                rethrow(ex)
            end
        end
    end
end


"""
Use Dtree to distribute the passed bounding boxes to multiple ranks for
processing. Within each rank, process the light sources in each of the
assigned boxes with multiple threads.
"""
function multi_node_joint_infer(all_boxes::Vector{BoundingBox},
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

    # thread barrier for ordering source processing
    nworkers = nthreads() - (mi.rundt ? 1 : 0)
    bar = setup_barrier(nworkers)

    # per-thread timing
    thread_timing = Array{InferTiming}(nthreads())

    config = Configs.Config()

    # thread function
    function process_boxes()
        tid = threadid()
        thread_timing[tid] = InferTiming()
        timing = thread_timing[tid]
        rng = MersenneTwister()

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
                atomic_cas!(cbox.state, BoxReady, BoxDone)
            end
            if !load_box(cbox, mi, all_boxes, stagedir;
                         primary_initialization=primary_initialization,
                         timing=timing)
                break
            end
            init_box(cbox, nworkers;
                     cyclades=cyclades_partition,
                     batch_size=batch_size,
                     use_fft=use_fft,
                     timing=timing)
            # process sources in the box
            tic()
            nbatches = length(cbox.sources_assignment[1])
            for iter = 1:niters
                for batch = 1:nbatches
                    # Shuffle the source assignments within each batch. This is
                    # disabled by default because it ruins the deterministic outcome
                    # required by the test cases.
                    # TODO: it isn't disabled by default?
                    if within_batch_shuffling
                        shuffle!(rng, cbox.sources_assignment[tid][batch])
                    end

                    #process_sources_elapsed_times = Vector{Float64}(nworkers)
                    #tic()
                    Log.debug("processing sources $(cbox.sources_assignment[tid][batch]), iter $iter, batch $batch")
                    process_sources(cbox.sources_assignment[tid][batch], cbox;
                                    use_fft=use_fft,
                                    within_batch_shuffling=within_batch_shuffling)
                    #process_sources_elapsed_times[tid] = toq()

                    # don't barrier on the last iteration
                    if !(iter == niters && batch == nbatches)
                        Log.debug("skipping barrier")
                        thread_barrier(bar)
                    end
                end
            end
            timing.opt_srcs += toq()

            # each thread writes results for its sources
            nsrcs = 0
            Log.debug("writing results")
            tic()
            for batch = 1:nbatches
                nsrcs = nsrcs + length(cbox.sources_assignment[tid][batch])
                for s in cbox.sources_assignment[tid][batch]
                    entry = cbox.catalog[cbox.target_sources[s]]
                    result = OptimizedSource(entry.thing_id,
                                             entry.objid,
                                             entry.pos[1],
                                             entry.pos[2],
                                             cbox.ea_vec[s].vp[1])
                    results_idx = source_counts[cbox.box_idx] + s
                    put!(all_results, results_idx, results_idx, [result])
                end
            end
            timing.ga_put += toq()
            timing.num_srcs += nsrcs
        end
    end

    if nthreads() == 1
        process_boxes()
    else
        ccall(:jl_threading_run, Void, (Any,), Core.svec(process_boxes))
        #ccall(:jl_threading_profile, Void, ())
    end
    Log.message("completed at $(Dates.now())")

    for i = 1:nthreads()
        add_timing!(timing, thread_timing[i])
    end

    tic()
    save_results(all_results, outdir)
    timing.write_results = toq()

    tic()
    finalize(mi.dt)
    timing.wait_done = toq()
    Log.message("synchronized ranks at $(Dates.now())")
end


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
        end
    end
end

