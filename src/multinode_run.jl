using Base.Threads
using Gasp

type BoxInfo
    sources::Vector{Int}
    catalog::Vector{CatalogEntry}
    neighbor_map::Vector{Vector{Int}}
    images::Vector{Image}
    lock::SpinLock
    curr_source::Int
    sources_done::Int

    BoxInfo() = new([], [], [], [], SpinLock(), 1, 0)
end

const NumConcurrentBoxes = 2::Int


"""
Use Dtree to distribute the passed bounding boxes to multiple nodes for
processing. Within each node, process the light sources in each of the
assigned boxes with multiple threads.
"""
function multi_node_infer(all_boxes::Vector{BoundingBox},
                          stagedir::String;
                          outdir=".",
                          infer_callback=one_node_single_infer,
                          primary_initialization=true,
                          timing=InferTiming())
    nwi = length(all_boxes)
    each = ceil(Int64, nwi / nnodes)

    if nodeid == 1
        nputs(nodeid, "running on $nnodes nodes")
        nputs(nodeid, "$nwi bounding boxes, ~$each per node")
    end

    # for concurrent box processing
    conc_boxes = [BoxInfo() for i=1:NumConcurrentBoxes]

    # per-thread timing
    ttimes = Array(InferTiming, nthreads())

    # results
    results = OptimizedSource[]
    results_lock = SpinLock()

    # create Dtree and get the initial allocation
    dt, is_parent = Dtree(nwi, 0.4)
    ni, (ci, li) = initwork(dt)
    ilock = SpinLock()
    rundt = runtree(dt)

    nputs(nodeid, "dtree: initial: $ni ($ci to $li)")

    function load_box(cbox::BoxInfo; timing=InferTiming())
        lock(cbox.lock)

        # another thread might have loaded a box here already
        if cbox.sources_done < length(cbox.sources)
            unlock(cbox.lock)
            return true
        end

        # determine which box to load next
        box_idx = 0
        while true
            lock(ilock)
            if li == 0
                unlock(ilock)
                ntputs(nodeid, tid, "dtree: out of work")
                return false
            end
            if ci > li
                ntputs(nodeid, tid, "dtree: consumed allocation (last was $li)")
                ni, (ci, li) = getwork(dt)
                unlock(ilock)
                ntputs(nodeid, tid, "dtree: $ni work items ($ci to $li)")
                continue
            end
            box_idx = ci
            ci = ci + 1
            unlock(ilock)
            break
        end

        box = all_boxes[box_idx]

        ntputs(nodeid, threadid(), "loading box $(box.ramin), $(box.ramax), $(box.decmin), $(box.decmax)")

        # load this box
        tic()
        rcfs = get_overlapping_fields(box, stagedir)
        timing.query_fids = timing.query_fids + toq()

        cbox.catalog, cbox.sources,
        cbox.neighbor_map, cbox.images = infer_init(rcfs, stagedir;
                                                    primary_initialization=primary_initialization,
                                                    timing=timing)
        cbox.curr_source = 1
        cbox.sources_done = 0
        unlock(cbox.lock)
        return true
    end

    # thread function to process sources in boxes issued by Dtree
    function process_boxes()
        tid = threadid()
        ttimes[tid] = InferTiming()
        times = ttimes[tid]

        # Dtree parent nodes reserve one thread to drive the tree
        if rundt && tid == 1
            ntputs(nodeid, tid, "dtree: running tree")
            while runtree(dt)
                Gasp.cpu_pause()
            end
        else
            # all other threads process sources in boxes
            curr_cbox = 1
            cbox = conc_boxes[curr_cbox]
            load_box(cbox; timing=times)
            while true
                # get the next source to process from the current box
                lock(cbox.lock)
                ts = cbox.curr_source
                cbox.curr_source = cbox.curr_source + 1
                unlock(cbox.lock)
ntputs(nodeid, tid, "source $ts of $(length(cbox.sources))")

                # if the current box is done, switch to the next box
                if ts > length(cbox.sources)
                    curr_cbox = curr_cbox + 1
                    if curr_cbox > NumConcurrentBoxes
                        curr_cbox = 1
                    end
                    cbox = conc_boxes[curr_cbox]
                    if load_box(cbox; timing=times)
                        continue
                    else
                        break
                    end
                end

                # process the source and record the result
                try
                    result = process_source(
                        Configs.Config(),
                        ts,
                        cbox.sources,
                        cbox.catalog,
                        cbox.neighbor_map,
                        cbox.images,
                    )

                    lock(results_lock)
                    push!(results, result)
                    unlock(results_lock)

                    lock(cbox.lock)
                    cbox.sources_done = cbox.sources_done + 1
                    unlock(cbox.lock)
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
        #save_results(outdir, box, results)
        times.write_results = toq()
    end

    tic()
    if nthreads() == 1
        process_boxes()
    else
        ccall(:jl_threading_run, Void, (Any,), Core.svec(process_boxes))
        ccall(:jl_threading_profile, Void, ())
    end

    tic()
    finalize(dt)
    timing.wait_done = toq()

    timing.opt_srcs = toq()
    timing.num_srcs = length(target_sources)
end

