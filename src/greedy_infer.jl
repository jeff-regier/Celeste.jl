using Base.Threads


immutable NoTasksAvailable <: Exception
end

immutable AllTasksDone <: Exception
end


immutable BoxTaskManager
    catalog::Vector{CatalogEntry}
    target_sources::Vector{Int64}
    neighbor_map::Vector{Vector{Int64}}
    images::Vector{Image}
    ea_vec::Vector{ElboArgs}
    vp_vec::Vector{VariationalParams{Float64}}
    cfg_vec::Vector{Config}
    target_source_variational_params::Vector{VariationalParams{Float64}}
    conflict_map::Vector{Vector{Int64}}
    task_in_progress::BitArray
    iters_left::Vector{Int64}
    lock::SpinLock
end


function BoxTaskManager(catalog::Vector{CatalogEntry},
                        target_sources::Vector{Int64},
                        neighbor_map::Vector{Vector{Int64}},
                        images::Vector{Image})
    # sort_targets! mutates the order of the both arguments. it needs to be done
    # first because everything that follows depends on their order
    sort_targets!(neighbor_map, target_sources)

    conflict_map = make_conflict_map(catalog, neighbor_map, target_sources)
    num_tasks = length(target_sources)
    # optimize sources without neighbors just once; others 3 times
    iters_left = [isempty(conflict_map[ts]) ? 1 : 3 for ts in 1:num_tasks]

    ea_vec, vp_vec, cfg_vec, target_source_variational_params =
            setup_vecs(num_tasks, target_sources, catalog)
    thread_initialize_sources_assignment::Vector{Vector{Vector{Int64}}} = partition_equally(n_threads, n_sources)
    initialize_elboargs_sources!(config, ea_vec, vp_vec, cfg_vec, thread_initialize_sources_assignment,
                                 catalog, target_sources, neighbor_map, images,
                                 target_source_variational_params)

    BoxTaskManager(catalog, target_sources, neighbor_map, images,
                   ea_vec, vp_vec, cfg_vec, target_source_variational_params,
                   conflict_map,
                   falses(num_tasks),
                   iters_left,
                   SpinLock())
end


function make_conflict_map(catalog, neighbor_map, target_sources)
    s_to_ts = fill(-1, length(catalog))

    for ts in 1:length(target_sources)
        s = target_sources[ts]
        s_to_ts[s] = ts
    end

    conflict_map = Vector{Vector{Int64}}(length(target_sources))

    for ts in 1:length(target_sources)
        conflict_map[ts] = Int64[]
        for s in neighbor_map[ts]
            ts2 = s_to_ts[s]
            if ts2 != -1
                append!(conflict_map[ts], ts2)
            end
        end
    end

    conflict_map
end


# Sort target_sources by the number of neighbors, descending, so those
# without neighbors are at the end of the list (they never have to "block",
# so we want to save them as long as threads can be kept busy with other work)
function sort_targets!(neighbor_map, target_sources)
    permutation = sortperm(length.(neighbor_map), rev=true)
    neighbor_map[:] = neighbor_map[permutation]
    target_sources[:] = target_sources[permutation]
end


# available = not in progress & iters remaining & neighbors are "caught up"
function is_task_available(btm::BoxTaskManager, ts::Int64)
    if btm.task_in_progress[ts] || btm.task_iters_left[ts] == 0
        return false
    end
    for ts2 in btm.conflict_map[task_id]
        if btm.task_in_progress[ts2] || btm.iters_left[ts2] < btm.iters_left[ts]
            return false
        end
    end
    return true
end


function get_task(btm::BoxTaskManager)
    lock!(btm.lock)

    task_id = 1
    while task_id <= num_tasks
        if is_task_available(btm, task_id)
            break
        elseif task_id == num_tasks
            unlock!(btm.lock)
            num_left = sum(btm.iters_left)
            throw(num_left > 0 ? NoTasksAvailable() : AllTasksDone())
        else
            task_id += 1
        end
    end

    btm.task_in_progress[task_id] = true
    unlock!(btm.lock)
    return task_id
end


function record_task_done(btm::BoxTaskManager, task_id::Int64)
    lock!(l)
    task_iters_left[task_id] -= 1
    task_in_progress[task_id] = false
    unlock!(l)
end


function do_task(btm, ts)
    Log.info("greedy infer is processing $ts")
    try
        maximize!(btm.ea_vec[ts], btm.vp_vec[ts], btm.cfg_vec[ts])
    catch ex
        if is_production_run || nthreads() > 1
            Log.exception(ex)
        else
            rethrow(ex)
        end
    end
end


function extract_results(btm)
    results = OptimizedSource[]

    for i = 1:num_tasks
        entry = btm.catalog[btm.target_sources[i]]
        result = OptimizedSource(entry.thing_id,
                                 entry.objid,
                                 entry.pos[1],
                                 entry.pos[2],
                                 vp_vec[i][1])
        push!(results, result)
    end

    return results
end


function one_node_greedy_infer(ctni...)
    btm = BoxTaskManager(ctni...)

    function task_loop()
        while true
            try
                ts = get_task(btm)
                optimize_source(btm, ts)
                record_task_done(btm, ts)
            catch ex
                if isa(ex, NoTasksAvailable)
                    continue
                elif isa(ex, AllTasksDone)
                    break
                else
                    Log.exception(ex)
                end
            end
        end
    end

    if nthreads() == 1
        task_loop()
    else
        ccall(:jl_threading_run, Void, (Any,), Core.svec(task_loop))
        ccall(:jl_threading_profile, Void, ())
    end

    show_pixels_processed()

    return extract_results(btm)
end
