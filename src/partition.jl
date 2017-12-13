# Assigning sources to threads based on which sources overlap others.
# 

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
        pqueue = PriorityQueue(Base.Order.Forward, (i, 0) for i=1:n_threads)

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
