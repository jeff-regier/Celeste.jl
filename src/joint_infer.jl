import FITSIO
import JLD
using DataStructures

import ..Log
using ..Model
import ..SDSSIO
import ..Infer
import ..SDSSIO: RunCamcolField
import ..PSF
using ..ConstraintTransforms: ConstraintBatch, DEFAULT_CHUNK
using ..DeterministicVI
using ..DeterministicVI.NewtonMaximize: Config, maximize!


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
    Log.info("Starting Cyclades partitioning...")
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

    Log.info("Cyclades - Assigned sources: $(assigned_sources) vs correct number of sources: $(n_sources)")
    @assert assigned_sources == n_sources
    Log.info("Cyclades - Number of batches: $(n_total_batches)")
    Log.info("Finished Cyclades partitioning.  Elapsed time: $(toq()) seconds")

    for cur_batch = 1:length(collect(1:batch_size:n_sources))
        load_balance_for_batch = [length(thread_sources_assignment[t][cur_batch]) for t=1:n_threads]
        Log.info("Load balance for batch $(cur_batch) - $(load_balance_for_batch)")
    end

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
    Log.info("Starting basic source partitioning...")
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
    Log.info("Finished basic source partitioning. Elapsed time: $(toq()) seconds")
    thread_sources_assignment
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
                              use_fft::Bool=false,
                              batch_size::Int=400,
                              within_batch_shuffling::Bool=true,
                              n_iters::Int=3)
    # Seed random number generator to ensure the same results per run.
    srand(42)

    n_threads = nthreads()

    # Partition the sources
    n_sources = length(target_sources)
    Log.info("Optimizing $(n_sources) sources")

    local thread_sources_assignment::Vector{Vector{Vector{Int64}}}

    if cyclades_partition
        # Convert neighbormap to cyclades map (map from source_id -> [neighbor source_ids])
        cyclades_neighbor_map = Dict{Int64, Vector{Int64}}()
        for (index, neighbors) in enumerate(neighbor_map)
            source_id = target_sources[index]
            cyclades_neighbor_map[source_id] = neighbors
        end
        thread_sources_assignment = partition_cyclades(n_threads, target_sources, cyclades_neighbor_map, batch_size=batch_size)
    else
        thread_sources_assignment = partition_equally(n_threads, n_sources)
    end

    Log.info("Done assigning sources to threads for processing")

    # Pre allocate elbo args variational params
    target_source_variational_params::Dict{Int64, Array{Float64}} = Dict{Int64, Array{Float64}}()
    for target_source in target_sources
        cat = catalog[target_source]
        target_source_variational_params[target_source] = generic_init_source(cat.pos)
    end

    # configurations need to be persisted across calls to maximize! so
    # that location constraints don't shift from their initial position
    ea_vec::Vector{ElboArgs} = Vector{ElboArgs}(n_sources)
    cfg_vec::Vector{Config{DEFAULT_CHUNK,Float64}} = Vector{Config{DEFAULT_CHUNK,Float64}}(n_sources)

    # Initialize elboargs in parallel
    tic()
    thread_initialize_sources_assignment::Vector{Vector{Vector{Int64}}} = partition_equally(n_threads, n_sources)

    initialize_elboargs_sources!(config, ea_vec, cfg_vec, thread_initialize_sources_assignment,
                                 catalog, target_sources, neighbor_map, images,
                                 use_fft, target_source_variational_params)

    Log.info("Done preallocating array of elboargs. Elapsed time: $(toq())")

    # Process sources in parallel
    tic()

    process_sources!(images, ea_vec, cfg_vec,
                     thread_sources_assignment,
                     n_iters, use_fft, within_batch_shuffling)

    Log.info("Done fitting elboargs. Elapsed time: $(toq())")

    # Return add results to vector
    results = OptimizedSource[]

    for i = 1:n_sources
        entry = catalog[target_sources[i]]
        result = OptimizedSource(entry.thing_id,
                                 entry.objid,
                                 entry.pos[1],
                                 entry.pos[2],
                                 ea_vec[i].vp[1])
        push!(results, result)
    end

    results
end

# legacy wrapper
function one_node_joint_infer(catalog, target_sources, neighbor_map, images;
                              cyclades_partition::Bool=true,
                              use_fft::Bool=false,
                              batch_size::Int=400,
                              within_batch_shuffling::Bool=true,
                              n_iters::Int=3)
    one_node_joint_infer(
        Configs.Config(),
        catalog,
        target_sources,
        neighbor_map,
        images,
        cyclades_partition=cyclades_partition,
        use_fft=use_fft,
        batch_size=batch_size,
        within_batch_shuffling=within_batch_shuffling,
        n_iters=n_iters
    )
end

function initialize_elboargs_sources!(config::Configs.Config, ea_vec, cfg_vec,
                                      thread_initialize_sources_assignment,
                                      catalog, target_sources, neighbor_map, images,
                                      use_fft, target_source_variational_params)
    Threads.@threads for i in 1:nthreads()
        try
            for batch in 1:length(thread_initialize_sources_assignment[i])
                initialize_elboargs_sources_kernel!(config, ea_vec, cfg_vec,
                                                    thread_initialize_sources_assignment[i][batch],
                                                    catalog, target_sources, neighbor_map, images,
                                                    use_fft, target_source_variational_params)
            end
        catch ex
            Log.exception(ex)
            rethrow()
        end
    end
end


function initialize_elboargs_sources_kernel!(config::Configs.Config, ea_vec, cfg_vec, sources,
                                             catalog, target_sources, neighbor_map, images,
                                             use_fft, target_source_variational_params)
    try
        for cur_source_index in sources
            entry_id = target_sources[cur_source_index]
            entry = catalog[target_sources[cur_source_index]]
            neighbor_ids = neighbor_map[cur_source_index]
            neighbors = catalog[neighbor_map[cur_source_index]]

            # TODO max: refactor this portion? It's reused in infer_source.
            cat_local = vcat([entry], neighbors)
            ids_local = vcat([entry_id], neighbor_ids)

            patches = Infer.get_sky_patches(images, cat_local)
            Infer.load_active_pixels!(config, images, patches)

            # Load vp with shared target source params, and also vp
            # that doesn't share target source params
            vp = Vector{Float64}[haskey(target_source_variational_params, x) ?
                                 target_source_variational_params[x] :
                                 catalog_init_source(catalog[x]) for x in ids_local]

            # Switch parameters based on whether or not we're using the fft method
            if use_fft
                ea, _ = initialize_fft_elbo_parameters(images,
                                                       vp,
                                                       patches,
                                                       [1],
                                                       use_raw_psf=false,
                                                       allocate_fsm_mat=true)
            else
                ea = ElboArgs(images, vp, patches, [1])
            end

            ea_vec[cur_source_index] = ea
            cfg_vec[cur_source_index] = Config(ea)
        end
        return nothing
    catch ex
        if is_production_run || nthreads() > 1
            Log.error(string(ex))
        else
            rethrow(ex)
        end
    end
end

function process_sources!(images::Vector{Model.Image},
                          ea_vec::Vector{ElboArgs},
                          cfg_vec::Vector{Config{DEFAULT_CHUNK,Float64}},
                          thread_sources_assignment::Vector{Vector{Vector{Int64}}},
                          n_iters::Int,
                          use_fft::Bool,
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
                    process_sources_kernel!(images, ea_vec, cfg_vec,
                                            thread_sources_assignment[i::Int][batch::Int],
                                            iter, use_fft, within_batch_shuffling)
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

# Process partition of sources. Multiple threads call this function in parallel.
function process_sources_kernel!(images::Vector{Model.Image},
                                 ea_vec::Vector{ElboArgs},
                                 cfg_vec::Vector{Config{DEFAULT_CHUNK,Float64}},
                                 source_assignment::Vector{Int64},
                                 iter::Int,
                                 use_fft::Bool,
                                 within_batch_shuffling::Bool)
    try
        # Shuffle the source assignments within each batch of each process.
        # This is disabled by default because it ruins the deterministic outcome
        # required by the test cases.
        if within_batch_shuffling
            shuffle!(source_assignment)
        end
        for i in source_assignment
            ea, cfg = ea_vec[i], cfg_vec[i]
            if use_fft
                f = FFTElboFunction(load_fsm_mat(ea, images; use_raw_psf=true))
            else
                f = DeterministicVI.elbo
            end
            maximize!(f, ea, cfg)
        end
    catch ex
        if is_production_run || nthreads() > 1
            Log.error(string(ex))
        else
            rethrow(ex)
        end
    end
end
