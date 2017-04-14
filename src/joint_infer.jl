import FITSIO
import JLD
using DataStructures

import ..Log
using ..Model
import ..SDSSIO
import ..Infer
import ..SDSSIO: RunCamcolField
import ..PSF

using ..DeterministicVI
using ..DeterministicVI.ConstraintTransforms: ConstraintBatch, DEFAULT_CHUNK
using ..DeterministicVI.ElboMaximize: Config, maximize!

const PeakFlops = false
gdlog = nothing

# 20 minute threshold
const BoxMaxThreshold = convert(UInt64, 20*60*1e9)::UInt64
const OneMinute = 0x0000000df8475800 # in nanoseconds

type BoxKillSwitch <: Function
    started::UInt64
    numfin_threshold::Int64
    numfin::Atomic{Int64}
    finished::Vector{UInt64}
    lastfin::Atomic{Int64}
    killed::Bool
    messaged::Bool

    BoxKillSwitch() = new(0, ceil(Int64, nthreads()/2), Atomic{Int64}(0),
                          fill(typemax(UInt64), nthreads()),
                          Atomic{Int64}(0), false, false)
end

if PeakFlops
function (f::BoxKillSwitch)(x)
    now_ns = time_ns()
    if f.started == 0
        # benign race here
        f.started = now_ns
        return false
    end
    intv = now_ns - f.started
    if intv > BoxMaxThreshold
        # and here
        return f.killed = true
    end
    if div(intv, OneMinute) > 0
        f.started = now_ns
        n_active = 0
        n_inactive = 0
        for elbo_vars in DeterministicVI.ElboMaximize.ELBO_VARS_POOL
            n_active += elbo_vars.active_pixel_counter[]
            n_inactive += elbo_vars.inactive_pixel_counter[]
        end
        message(gdlog, "pixel visits: ($n_active,$n_inactive)")
    end
    return f.killed
end
else # !PeakFlops
function (f::BoxKillSwitch)(x)
    now_ns = time_ns()
    if f.started == 0
        # benign race here
        f.started = now_ns
        return false
    end
    intv = now_ns - f.started
    if intv > BoxMaxThreshold
        # and here
        return f.killed = true
    end
#=
    if f.numfin[] >= f.numfin_threshold
        lastfinat = f.finished[f.lastfin[]]
        if now_ns - lastfinat > floor(Int64, (lastfinat - f.started) * 0.25)
            #f.killed = true
            if !f.messaged
                f.messaged = true
                ttms = [f.finished[i] == typemax(UInt64) ? "" :
                            @sprintf("%d:%.3f ", i, (f.finished[i] - f.started) / 1e6)
                        for i in 1:length(f.finished)]
                Log.message("imbalance threshold: $(ttms...)ms")
            end
        end
=#
    return f.killed
end
end

function ks_reset_finished(ks::BoxKillSwitch)
    ks.numfin[] = 0
    fill!(ks.finished, typemax(UInt64))
    ks.lastfin[] = 0
    ks.messaged = false
end


function union_find!(i, components_tree)
    root = i
    while components_tree[i] != i
        i = components_tree[i]
    end

    while root != i
        root2 = components_tree[root]
        components_tree[root] = i
        root = root2
    end
    return i
end

"""
Computes connected components given a conflict graph, and the range of sources
to process. Writes to the components argument.

- source_start - start source
- end_source - end source (inclusively processed)
- sources - the actual source ids to find connected components for
    (indexed by source_start and source_end)
- neighbor_map - conflict graph of sources
- components_tree - the components array to do union find on.
- components - Dict{Int64, Vector{Int64}} dictionary from cc_id to
              target_source index in that component
- src_to_target_index - a mapping from light source id -> index
            within the target_sources array. This is required
            since we need to write the index within target sources
            to the output, rather than the source id itself.
Returns:
- Nothing
"""
function compute_connected_components(source_start, source_end, sources, neighbor_map,
                                      components_tree, components, src_to_target_index)
    # Construct reverse map from source_id -> index in source. Only do this for
    # the specified portion we are computing connected components.
    source_to_index = Dict{Int64, Int64}()
    index_to_source = Dict{Int64, Int64}()
    for i = source_start:source_end
        source_to_index[sources[i]] = i-source_start+1
        index_to_source[i-source_start+1] = sources[i]
    end

    for i = source_start:source_end
        components_tree[i-source_start+1] = i-source_start+1
    end

    # Run union find algorithm to find connected components.
    for i = source_start:source_end
        target = union_find!(i-source_start+1, components_tree)
        for neighbor in neighbor_map[index_to_source[i-source_start+1]]
            if haskey(source_to_index, neighbor)
                neighbor_idx = source_to_index[neighbor]
                component_source = union_find!(neighbor_idx, components_tree)
                components_tree[component_source] = target
            end

        end
    end

    # Write to components dictionary.
    for i = source_start:source_end
        index = union_find!(i-source_start+1, components_tree)
        if !haskey(components, index)
            components[index] = Vector{Int64}()
        end
        push!(components[index], src_to_target_index[index_to_source[i-source_start+1]])
    end
end

"""
Partitions sources via the cyclades algorithm.

- n_threads - number of threads to which to distribute sources
- target_sources - array of target sources. Elements should match keys of neighbor_map.
- neighbor_map - graph of connections of sources

Returns:
- An array of vectors representing the workload of each thread ([thread][batch][sources(indices)])
"""
function partition_cyclades(n_threads, target_sources, neighbor_map; batch_size=60)
    #Log.info("Starting Cyclades partitioning...")
    tic()

    n_sources = length(target_sources)
    n_total_batches = convert(Int64, ceil(n_sources / batch_size))

    # Construct src_to_indx map
    src_to_indx = Dict{Int64, Int64}()
    for i=1:n_sources
        src_to_indx[target_sources[i]] = i
    end

    # The final workload distribution.
    thread_sources_assignment = Vector{Vector{Vector{Int64}}}(n_threads)
    for thread = 1:n_threads
        thread_sources_assignment[thread] = Vector{Vector{Int64}}(n_total_batches)
        for batch = 1:n_total_batches
            thread_sources_assignment[thread][batch] = Vector{Int64}()
        end
    end

    # First shuffle the sources. Note Cyclades is serially equivalent
    # to this permutation of sources.
    sources = [x for x in keys(neighbor_map)]
    shuffle!(sources)

    # Process batch_size number of sources at a time.
    # TODO max - parallelize everything below (particularly CC computation)

    # The components tree is for union-find.
    components_tree = [Vector{Int64}(batch_size) for i=1:n_threads]

    # We have n_total_batches components, where each component is a dictionary
    # from the component id, to a list of sources in that component
    components = [Dict{Int64, Vector{Int64}}() for i=1:n_total_batches]
    sources_in_components = 0
    for (cur_batch, source_index) in enumerate(collect(1:batch_size:n_sources))
        source_start = source_index
        source_end = min(source_index+batch_size-1, n_sources)
        @assert source_end - source_start + 1 <= batch_size

        # TODO max - parallelize this
        compute_connected_components(source_start, source_end, sources, neighbor_map,
                                     components_tree[1],
                                     components[cur_batch],
                                     src_to_indx)
    end

    assigned_sources = 0
    # Load balance the connected components within each batch into thread_sources_assignment.
    for (cur_batch, cur_batch_component) in enumerate(components)
        # Priority queue for load balancing.
        pqueue = PriorityQueue([i for i=1:n_threads],
                               [0 for i=1:n_threads],
                               Base.Order.Forward)

        # Assign non-conflicting group of sources to different threads
        for (component_group_id, sources_of_component) in cur_batch_component
            least_loaded_thread = peek(pqueue)[1]
            for source_to_assign in sources_of_component
                push!(thread_sources_assignment[least_loaded_thread][cur_batch], source_to_assign)
                assigned_sources += 1
            end
            pqueue[least_loaded_thread] += length(sources_of_component)
        end
    end

    #Log.info("Cyclades - Assigned sources: $(assigned_sources) vs correct number of sources: $(n_sources)")
    @assert assigned_sources == n_sources
    #Log.info("Cyclades - Number of batches: $(n_total_batches)")
    #Log.info("Finished Cyclades partitioning.  Elapsed time: $(toq()) seconds")

    for cur_batch = 1:length(collect(1:batch_size:n_sources))
        load_balance_for_batch = [length(thread_sources_assignment[t][cur_batch]) for t=1:n_threads]
        #Log.info("Load balance for batch $(cur_batch) - $(load_balance_for_batch)")
    end

    thread_sources_assignment
end

"""
Partitions sources via the cyclades algorithm.

- target_sources - array of target sources. Elements should match keys of neighbor_map.
- neighbor_map - graph of connections of sources

Returns:
- An array of vectors representing the workload of each thread ([thread][batch][sources(indices)])
"""
function partition_cyclades_dynamic(target_sources, neighbor_map; batch_size=60)
    #Log.info("Starting Cyclades partitioning...")
    tic()

    n_sources = length(target_sources)
    n_total_batches = convert(Int64, ceil(n_sources / batch_size))

    # Construct src_to_indx map
    src_to_indx = Dict{Int64, Int64}()
    for i=1:n_sources
        src_to_indx[target_sources[i]] = i
    end

    # The final workload distribution.
    thread_sources_assignment = Vector{Vector{Vector{Int64}}}(n_total_batches)
    for batch = 1:n_total_batches
        thread_sources_assignment[batch] = Vector{Vector{Int64}}()
    end

    # First shuffle the sources. Note Cyclades is serially equivalent
    # to this permutation of sources.
    sources = [x for x in keys(neighbor_map)]
    shuffle!(sources)

    # Process batch_size number of sources at a time.
    # TODO max - parallelize everything below (particularly CC computation)

    # The components tree is for union-find.
    components_tree = Vector{Int64}(batch_size)

    # We have n_total_batches components, where each component is a dictionary
    # from the component id, to a list of sources in that component
    components = [Dict{Int64, Vector{Int64}}() for i=1:n_total_batches]
    sources_in_components = 0
    for (cur_batch, source_index) in enumerate(collect(1:batch_size:n_sources))
        source_start = source_index
        source_end = min(source_index+batch_size-1, n_sources)
        @assert source_end - source_start + 1 <= batch_size

        # TODO max - parallelize this
        compute_connected_components(source_start, source_end, sources, neighbor_map,
                                     components_tree,
                                     components[cur_batch],
                                     src_to_indx)
    end

    assigned_sources = 0
    # Add sources to sources assignment
    for (cur_batch, cur_batch_component) in enumerate(components)
        for (component_group_id, sources_of_component) in cur_batch_component
            push!(thread_sources_assignment[cur_batch], copy(sources_of_component))
        assigned_sources += length(sources_of_component)
        end
    end

    #Log.info("Cyclades - Assigned sources: $(assigned_sources) vs correct number of sources: $(n_sources)")
    @assert assigned_sources == n_sources
    #Log.info("Cyclades - Number of batches: $(n_total_batches)")
    #Log.info("Finished Cyclades partitioning.  Elapsed time: $(toq()) seconds")

    #Log.info("Assigned sources: $(thread_sources_assignment)")

    thread_sources_assignment
end


"""
Partitions sources across threads equally.

- n_threads - number of threads to which to distribute processing the sources.
- n_sources - the number of total sources to process.

Returns:
- An array of vectors representing the workload of each thread ([thread][batch][sources]).
  Note for partition equally, there is only 1 batch.

"""
function partition_equally(n_threads, n_sources)
    #Log.info("Starting basic source partitioning...")
    tic()
    n_sources_per_thread = floor(Int64, n_sources / n_threads)
    thread_sources_assignment = Vector{Vector{Vector{Int64}}}(n_threads)
    n_sources_assigned = 0
    for thread = 1:n_threads
        start_source = (thread-1) * n_sources_per_thread
        end_source = thread * n_sources_per_thread
        if thread == n_threads
            end_source = n_sources
        end
        # Only 1 batch for partitioning equally
        thread_sources_assignment[thread] = Vector{Vector{Int64}}(1)
        thread_sources_assignment[thread][1] = Vector{Int64}(end_source-start_source)
        for source_assignment = start_source:end_source-1
            thread_sources_assignment[thread][1][source_assignment-start_source+1] = source_assignment+1
            n_sources_assigned += 1
        end
    end
    @assert n_sources_assigned == n_sources
    #Log.info("Finished basic source partitioning. Elapsed time: $(toq()) seconds")
    thread_sources_assignment
end


"""
Return a Cyclades partitioning or an equal partitioning of the target
sources.
"""
function partition_box(npartitions::Int, target_sources::Vector{Int},
                       neighbor_map::Vector{Vector{Int}}, ea_vec;
                       cyclades_partition=true,
                       batch_size=7000)
    if cyclades_partition
        cyclades_neighbor_map = Dict{Int64,Vector{Int64}}()
        for (index, neighbors) in enumerate(neighbor_map)
            cyclades_neighbor_map[target_sources[index]] = neighbors
        end
        #return partition_cyclades(npartitions, target_sources,
        #                          cyclades_neighbor_map,
        #                          batch_size=batch_size)
        #return partition_cyclades_dynamic(target_sources,
        #                                  cyclades_neighbor_map,
        #                                  batch_size=batch_size)
	return partition_cyclades_dynamic_auto_batchsize(target_sources, cyclades_neighbor_map, ea_vec)
    else
        return partition_equally(npartitions, length(target_sources))
    end
end

function estimate_time(patches)
    sum(sum(p.active_pixel_bitmap) for p in patches)
end

function load_balance_across_threads(n_threads, times)
    ts = [0 for i=1:n_threads]
    for t in times
        minimum,index_min = findmin(ts)
        ts[index_min] += t
    end
    return ts
end

"""
Partitions sources via the cyclades algorithm. Finds the batch size which returns most balanced CC distribution.
- target_sources - array of target sources. Elements should match keys of neighbor_map.
- neighbor_map - graph of connections of sources
"""
function partition_cyclades_dynamic_auto_batchsize(target_sources, neighbor_map, ea_vec)
    n_threads = nthreads()
    
    # Sample batch sizes at intervals
    n_to_sample = 100
    stepsize = max(1, trunc(Int, length(target_sources) / n_to_sample))

    # Best load imbalance of the batch sizes to test
    best_score = Inf
    best_result = Inf
    best_batch_size = -Inf

    for batch_size_to_use = 1 : stepsize : length(target_sources)+1
        ccs = partition_cyclades_dynamic(target_sources, neighbor_map, batch_size=batch_size_to_use)
        score = 0
        for batch in ccs
            # Find average load imbalance within the batch as a percentage
	    times = [sum([estimate_time(ea_vec[source_index].patches) for source_index in component]) for component in batch]
            times = load_balance_across_threads(n_threads, times)
            estimated_imbalance = mean(maximum(times) - times) 
	    score += estimated_imbalance
        end
        if score <= best_score
            best_result = ccs
            best_score = score
            best_batch_size = batch_size_to_use
        end
    end
    for batch in best_result
        sizes = [length(x) for x in batch]
    end
    best_result
end

"""
Pre-allocate elbo args variational parameters; configurations need to
be persisted across calls to `maximize!()` so that location constraints
don't shift from their initial position.
"""
function setup_vecs(n_sources::Int, target_sources::Vector{Int},
                    catalog::Vector{CatalogEntry})
    ts_vp = Dict{Int64, Array{Float64}}()
    for ts in target_sources
        cat = catalog[ts]
        ts_vp[ts] = generic_init_source(cat.pos)
    end

    ea_vec = Vector{ElboArgs}(n_sources)
    vp_vec = Vector{VariationalParams{Float64}}(n_sources)
    cfg_vec = Vector{Config{DEFAULT_CHUNK,Float64}}(n_sources)

    return ea_vec, vp_vec, cfg_vec, ts_vp
end


"""
Like one_node_infer, uses multiple threads on one node to fit the Celeste
model over numerous iterations.

catalog - the catalog of light sources
target_sources - light sources to optimize
neighbor_map - ligh_source index -> neighbor light_source id

cyclades_partition - use the cyclades algorithm to partition into non conflicting batches for updates.
batch_size - size of a single batch of sources for updates
within_batch_shuffling - whether or not to process sources within a batch randomly
joint_inference_terminate - whether to terminate once sources seem to be stable
n_iters - number of iterations to optimize. 1 iteration optimizes a full pass over target
          sources if optimize_fixed_iters=true.

Returns:

- Vector of OptimizedSource results
"""
function one_node_joint_infer(config::Configs.Config, catalog, target_sources, neighbor_map, images;
                              cyclades_partition::Bool=true,
                              batch_size::Int=7000,
                              within_batch_shuffling::Bool=true,
                              n_iters::Int=3,
                              timing=InferTiming())
    # Seed random number generator to ensure the same results per run.
    srand(42)

    n_threads = nthreads()

    # Partition the sources
    n_sources = length(target_sources)
    Log.info("Optimizing $(n_sources) sources")

    #Log.info(batched_connected_components)

    Log.info("Done assigning sources to threads for processing")

    ea_vec, vp_vec, cfg_vec, target_source_variational_params =
            setup_vecs(n_sources, target_sources, catalog)

    ks = BoxKillSwitch()

    # Initialize elboargs in parallel
    tic()
    thread_initialize_sources_assignment::Vector{Vector{Vector{Int64}}} = partition_equally(n_threads, n_sources)

    initialize_elboargs_sources!(config, ea_vec, vp_vec, cfg_vec, thread_initialize_sources_assignment,
                                 catalog, target_sources, neighbor_map, images,
                                 target_source_variational_params;
                                 termination_callback=ks)
    timing.init_elbo = toq()

    #thread_sources_assignment = partition_box(n_threads, target_sources,
    #                                  neighbor_map;
    #                                  cyclades_partition=cyclades_partition,
    #                                  batch_size=batch_size)
    batched_connected_components = partition_box(n_threads, target_sources,
                                                 neighbor_map, ea_vec;
                                                 cyclades_partition=cyclades_partition,
                                                 batch_size=batch_size)

    # Process sources in parallel
    tic()

    #process_sources!(images, ea_vec, vp_vec, cfg_vec,
    #                 thread_sources_assignment,
    #                 n_iters, within_batch_shuffling)
    process_sources_dynamic!(images, ea_vec, vp_vec, cfg_vec,
                             batched_connected_components,
                             n_iters, within_batch_shuffling;
                             kill_switch=ks)

    timing.opt_srcs = toq()
    timing.num_srcs = n_sources

    # Return add results to vector
    results = OptimizedSource[]

    if ks.killed
        Log.message("optimization terminated due to timeout")
    else
        for i = 1:n_sources
            entry = catalog[target_sources[i]]
            result = OptimizedSource(entry.thing_id,
                                     entry.objid,
                                     entry.pos[1],
                                     entry.pos[2],
                                     vp_vec[i][1])
            push!(results, result)
        end
    end

    show_pixels_processed()

    results
end

# legacy wrapper
function one_node_joint_infer(catalog, target_sources, neighbor_map, images;
                              cyclades_partition::Bool=true,
                              batch_size::Int=7000,
                              within_batch_shuffling::Bool=true,
                              n_iters::Int=3,
                              timing=InferTiming())
    one_node_joint_infer(
        Configs.Config(),
        catalog,
        target_sources,
        neighbor_map,
        images,
        cyclades_partition=cyclades_partition,
        batch_size=batch_size,
        within_batch_shuffling=within_batch_shuffling,
        n_iters=n_iters,
        timing=timing
    )
end

function initialize_elboargs_sources!(config::Configs.Config, ea_vec, vp_vec, cfg_vec,
                                      thread_initialize_sources_assignment,
                                      catalog, target_sources, neighbor_map, images,
                                      target_source_variational_params;
                                      termination_callback=nothing)
    Threads.@threads for i in 1:nthreads()
        try
            for batch in 1:length(thread_initialize_sources_assignment[i])
                for source_index in thread_initialize_sources_assignment[i][batch]
                    init_elboargs(config, source_index, catalog, target_sources,
                                  neighbor_map, images, ea_vec, vp_vec, cfg_vec,
                                  target_source_variational_params;
                                  termination_callback=termination_callback)
                end
            end
        catch ex
            Log.exception(ex)
            rethrow()
        end
    end
end


"""
Initialize elbo args for the specified target source.
"""
function init_elboargs(config::Configs.Config,
                       ts::Int,
                       catalog::Vector{CatalogEntry},
                       target_sources::Vector{Int},
                       neighbor_map::Vector{Vector{Int}},
                       images::Vector{Image},
                       ea_vec::Vector{ElboArgs},
                       vp_vec::Vector{VariationalParams{Float64}},
                       cfg_vec::Vector{Config{DEFAULT_CHUNK,Float64}},
                       ts_vp::Dict{Int64,Array{Float64}};
                       termination_callback=nothing)
    try
        entry_id = target_sources[ts]
        entry = catalog[entry_id]
        neighbor_ids = neighbor_map[ts]
        neighbors = catalog[neighbor_ids]
        cat_local = vcat([entry], neighbors)
        ids_local = vcat([entry_id], neighbor_ids)

        patches = Infer.get_sky_patches(images, cat_local)
        Infer.load_active_pixels!(config, images, patches)
        # Load vp with shared target source params, and also vp
        # that doesn't share target source params
        vp = Vector{Float64}[haskey(ts_vp, x) ?
                             ts_vp[x] :
                             catalog_init_source(catalog[x])
                             for x in ids_local]
        ea = ElboArgs(images, patches, [1])

        ea_vec[ts] = ea
        vp_vec[ts] = vp
        cfg_vec[ts] = Config(ea, vp;
                termination_callback=termination_callback)
    catch exc
        if is_production_run || nthreads() > 1
            Log.exception(exc)
        else
            rethrow()
        end
    end
end


function process_sources!(images::Vector{Model.Image},
                          ea_vec::Vector{ElboArgs},
                          vp_vec::Vector{VariationalParams{Float64}},
                          cfg_vec::Vector{Config{DEFAULT_CHUNK,Float64}},
                          thread_sources_assignment::Vector{Vector{Vector{Int64}}},
                          n_iters::Int,
                          within_batch_shuffling::Bool)
    n_threads::Int = nthreads()
    n_batches::Int = length(thread_sources_assignment[1])
    for iter in 1:n_iters
        # Process every batch of every iteration. We do the batches on the outside
        # Since there is an implicit barrier after the inner threaded for loop below.
        # We want this barrier because there may be conflict _between_ Cyclades batches.
        for batch in 1:n_batches
            process_sources_elapsed_times::Vector{Float64} = Vector{Float64}(n_threads)
            # Process every batch of every iteration with n_threads
            Threads.@threads for i in 1:n_threads
                try
                    tic()
                    process_sources_kernel!(ea_vec, vp_vec, cfg_vec,
                                            thread_sources_assignment[i::Int][batch::Int],
                                            within_batch_shuffling)
                    process_sources_elapsed_times[i::Int] = toq()
                catch exc
                    Log.exception(exc)
                    rethrow()
                end
            end
            Log.info("Batch $(batch) - $(process_sources_elapsed_times)")
        end
    end
end

function process_sources_dynamic!(images::Vector{Model.Image},
                                  ea_vec::Vector{ElboArgs},
                                  vp_vec::Vector{VariationalParams{Float64}},
                                  cfg_vec::Vector{Config{DEFAULT_CHUNK,Float64}},
                                  thread_sources_assignment::Vector{Vector{Vector{Int64}}},
                                  n_iters::Int,
                                  within_batch_shuffling::Bool;
                                  kill_switch=nothing)
    Log.info("Processing with dynamic connected components load balancing")

    n_threads::Int = nthreads()
    n_batches::Int = length(thread_sources_assignment)

    l = SpinLock()

    total_idle_time = 0
    total_sum_of_thread_times = 0

    for iter in 1:n_iters

        # Process every batch of every iteration. We do the batches on the outside
        # Since there is an implicit barrier after the inner threaded for loop below.
        # We want this barrier because there may be conflict _between_ Cyclades batches.
        for batch in 1:n_batches

            connected_components_index = 1
            process_sources_elapsed_times::Vector{Float64} = Vector{Float64}(n_threads)

            # Process every batch of every iteration with n_threads
            Threads.@threads for i in 1:n_threads
                try
                    tic()
                    while true
                        cc_index = -1

                        #ccall(:jl_,Void,(Any,), "this is thread number $(Threads.threadid()) $(cc_index)")

                        lock(l)
                        cc_index = connected_components_index
                        connected_components_index += 1
                        unlock(l)

                        if cc_index > length(thread_sources_assignment[batch::Int])
                            break
                        end

                        #ccall(:jl_,Void,(Any,), "this is thread number $(Threads.threadid()) $(cc_index) popped")

                        process_sources_kernel!(ea_vec, vp_vec, cfg_vec,
                                                thread_sources_assignment[batch::Int][cc_index],
                                                within_batch_shuffling)
                        if kill_switch != nothing && kill_switch.killed
                            break
                        end
                    end
                    process_sources_elapsed_times[i::Int] = toq()
                catch exc
                    Log.exception(exc)
                    rethrow()
                end
                if kill_switch != nothing
                    tid = threadid()
                    kill_switch.finished[tid] = time_ns()
                    kill_switch.lastfin[] = tid
                    atomic_add!(kill_switch.numfin, 1)
                end
            end
            Log.info("Batch $(batch) - $(process_sources_elapsed_times)")
	    avg_thread_idle_time = mean(maximum(process_sources_elapsed_times) - process_sources_elapsed_times)
	    maximum_thread_time = maximum(process_sources_elapsed_times)
            idle_percent = 100.0 * (avg_thread_idle_time / maximum_thread_time)
            Log.info("Batch $(batch) avg threads idle: $(round(Int, idle_percent))% ($(avg_thread_idle_time) / $(maximum_thread_time))")
	    total_idle_time += sum(maximum(process_sources_elapsed_times) - process_sources_elapsed_times)
            total_sum_of_thread_times += sum(process_sources_elapsed_times)
            if kill_switch != nothing
                if kill_switch.killed
                    break
                else
                    ks_reset_finished(kill_switch)
                end
            end
        end
        if kill_switch != nothing && kill_switch.killed
            break
        end
    end
    Log.info("Total idle time: $(round(Int, total_idle_time)), Total sum of threads times: $(round(Int, total_sum_of_thread_times))")
end

# Process partition of sources. Multiple threads call this function in parallel.
function process_sources_kernel!(ea_vec::Vector{ElboArgs},
                                 vp_vec::Vector{VariationalParams{Float64}},
                                 cfg_vec::Vector{Config{DEFAULT_CHUNK,Float64}},
                                 source_assignment::Vector{Int64},
                                 within_batch_shuffling::Bool)
    try
        # Shuffle the source assignments within each batch of each process.
        # This is disabled by default because it ruins the deterministic outcome
        # required by the test cases.
        if within_batch_shuffling
            shuffle!(source_assignment)
        end
	
	#tic()
        for i in source_assignment
            maximize!(ea_vec[i], vp_vec[i], cfg_vec[i])
            if cfg_vec[i].optim_options.callback.killed
                break
            end
        end
	#ccall(:jl_,Void,(Any,), "this is thread number $(Threads.threadid()) took $(toq()) seconds to process batch with $(length(source_assignment)) sources")
    catch ex
        if is_production_run || nthreads() > 1
            Log.exception(ex)
        else
            rethrow(ex)
        end
    end
end

function get_pixels_processed()
    n_active = 0
    n_inactive = 0
    for elbo_vars in DeterministicVI.ElboMaximize.ELBO_VARS_POOL
        n_active += elbo_vars.active_pixel_counter[]
        n_inactive += elbo_vars.inactive_pixel_counter[]
        elbo_vars.active_pixel_counter[] = 0
        elbo_vars.inactive_pixel_counter[] = 0
    end
    return (n_active, n_inactive)
end

# Count and display the number of active and inactive pixels processed.
function show_pixels_processed()
    n_active, n_inactive = get_pixels_processed()
    Log.message("$(Time(now())): (active,inactive) pixels processed: \($n_active,$n_inactive\)")
end

