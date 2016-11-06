import FITSIO
import JLD

import ..Log
using ..Model
import ..SDSSIO
import ..Infer
import ..SDSSIO: RunCamcolField
import ..PSF

using ..DeterministicVI

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
TODO max - refactor into different file.

- source_start - start source
- end_source - end source (inclusively processed)
- sources - the actual source ids to find connected components for (indexed by source_start and source_end)
- neighbor_map - conflict graph of sources
- components_tree - the components array to do union find on.
- components - Dict{Int64, Vector{Int64}} dictionary from cc_id to target_source index in that component
- src_to_target_index - a mapping from light source id -> index within the target_sources array. This is required
                        since we need to write the index within target sources to the output, rather than the source id itself.

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
TODO max - refactor into different source file (E.G: Cyclades.jl)

- nprocthreads - number of threads to which to distribute sources
- target_sources - array of target sources. Elements should match keys of neighbor_map.
- neighbor_map - graph of connections of sources

Returns:
- An array of vectors representing the workload of each thread ([thread][batch][sources(indices)])
"""
function partition_cyclades(nprocthreads, target_sources, neighbor_map; batch_size=60)
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
    thread_sources_assignment = Array(Vector{Vector{Int64}}, nprocthreads)
    for thread = 1:nprocthreads
        thread_sources_assignment[thread] = Vector{Vector{Int64}}(n_total_batches)
        for batch = 1:n_total_batches
            thread_sources_assignment[thread][batch] = Vector{Int64}()
        end
    end    

    # First shuffle the sources. Note Cyclades is serially equivalent
    # to this permutation of sources.
    sources = [x for x in keys(neighbor_map)]
    #shuffle!(sources)

    # Process batch_size number of sources at a time.
    # TODO max - parallelize everything below (particularly CC computation)

    # The components tree is for union-find.
    components_tree = [Array(Int64, batch_size) for i=1:nprocthreads]

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
        pqueue = Base.Collections.PriorityQueue([i for i=1:nprocthreads],
                                                [0 for i=1:nprocthreads],
                                                Base.Order.Forward)
        
        # Assign non-conflicting group of sources to different threads
        for (component_group_id, sources_of_component) in cur_batch_component
            least_loaded_thread = Base.Collections.peek(pqueue)[1]
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
    thread_sources_assignment
end

"""
Partitions sources across threads equally.
TODO max - refactor into different source file (E.G: Cyclades.jl)

- nprocthreads - number of threads to which to distribute processing the sources.
- n_sources - the number of total sources to process.

Returns:
- An array of vectors representing the workload of each thread ([thread][batch][sources]). 
  Note for partition equally, there is only 1 batch.

"""
function partition_equally(nprocthreads, n_sources)
    Log.info("Starting basic source partitioning...")
    tic()
    n_sources_per_thread = floor(Int64, n_sources / nprocthreads)
    thread_sources_assignment = Array(Vector{Vector{Int64}}, nprocthreads)
    n_sources_assigned = 0
    for thread = 1:nprocthreads
        start_source = (thread-1) * n_sources_per_thread
        end_source = thread * n_sources_per_thread
        if thread == nprocthreads
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
TODO max - refactor into different source file (E.G: Cyclades.jl)? Maybe also rename?

catalog - the catalog of light sources
target_sources - light sources to optimize
neighbor_map - ligh_source index -> neighbor light_source id

Returns:

- Dictionary of results, keyed by SDSS thing_id.
"""
function one_node_joint_infer(catalog, target_sources, neighbor_map, images;
                              cyclades_partition=true,
                              n_iters=10,
                              objid="")
    nprocthreads = nthreads()
    
    # Partition the sources
    n_sources = length(target_sources)
    Log.info("Optimizing $(n_sources) sources")
    if cyclades_partition
        # Convert neighbormap to cyclades map (map from source_id -> [neighbor source_ids])
        cyclades_neighbor_map = Dict{Int64, Vector{Int64}}()
        for (index, neighbors) in enumerate(neighbor_map)
            source_id = target_sources[index]
            cyclades_neighbor_map[source_id] = neighbors
        end
        thread_sources_assignment = partition_cyclades(nprocthreads, target_sources, cyclades_neighbor_map)
    else
        thread_sources_assignment = partition_equally(nprocthreads, n_sources)
    end

    Log.info("Done assigning sources to threads for processing")

    # Pre allocate elbo args variational params
    target_source_variational_params = Dict{Int64, Array{Float64}}()
    for target_source in target_sources
        cat = catalog[target_source]
        target_source_variational_params[target_source] = init_source(cat)
    end
    
    # Pre-allocate dictionary of elboargs, call it model.
    model = Array{ElboArgs}(n_sources)
    function initialize_elboargs_sources(sources)
        nputs(dt_nodeid, "Thread $(Threads.threadid()) allocating mem for $(length(sources)) sources")
        for cur_source_index in sources
            entry_id = target_sources[cur_source_index]
            entry = catalog[target_sources[cur_source_index]]
            neighbor_ids = neighbor_map[cur_source_index]
            neighbors = catalog[neighbor_map[cur_source_index]]

            # TODO max: refactor this portion? It's reused in infer_source.
            nputs(dt_nodeid, "Thread $(Threads.threadid()) allocating mem for source $(target_sources[cur_source_index]): objid=$(entry.objid)")
            cat_local = vcat(entry, neighbors)
            ids_local = vcat(entry_id, neighbor_ids)
            
            #vp = Vector{Float64}[init_source(ce) for ce in cat_local]
            vp = Vector{Float64}[haskey(target_source_variational_params, x) ? target_source_variational_params[x] : init_source(catalog[x]) for x in ids_local]
            patches, tile_source_map = Infer.get_tile_source_map(images, cat_local)
            ea = ElboArgs(images, vp, tile_source_map, patches, [1])
            Infer.fit_object_psfs!(ea, ea.active_sources)
            Infer.load_active_pixels!(ea)
            @assert length(ea.active_pixels) > 0
            model[cur_source_index] = ea
        end
    end

    # Initialize elboargs in parallel
    tic()
    thread_initialize_sources_assignment = partition_equally(nprocthreads, n_sources)
    
    Threads.@threads for i=1:nprocthreads
        for batch = 1:length(thread_initialize_sources_assignment[i])
            initialize_elboargs_sources(thread_initialize_sources_assignment[i][batch])
        end
    end

    Log.info("Done preallocating array of elboargs. Elapsed time: $(toq())")

    # Keep track of object values achieved per light source
    obj_values = Array{Float64}(length(target_sources))

    # Process partition of sources. Multiple threads call this function in parallel.
    function process_sources(source_assignment::Vector{Int64})
        for cur_source_indx in source_assignment
            cur_entry = catalog[target_sources[cur_source_indx]]
            nputs(dt_nodeid, "Thread $(Threads.threadid()) processing source $(target_sources[cur_source_indx]): objid = $(cur_entry.objid)")
            nputs(dt_nodeid, "Before: $(model[cur_source_indx].vp[1])")
            iter_count, obj_value, max_x, r = DeterministicVI.maximize_f(DeterministicVI.elbo, model[cur_source_indx], max_iters=10)
            nputs(dt_nodeid, "After: $(model[cur_source_indx].vp[1])")
            obj_values[cur_source_indx] = obj_value
        end
    end


    # Process sources in parallel using nprocthreads.
    tic()
    n_batches = length(thread_sources_assignment[1])
    for iter = 1:n_iters
        # Process every batch of every iteration. We do the batches on the outside
        # Since there is an implicit barrier after the inner threaded for loop below.
        # We want this barrier because there may be conflict _between_ Cyclades batches.
        for batch = 1:n_batches
            # Process every batch of every iteration with nprocthreads
            Threads.@threads for i = 1:nprocthreads
                process_sources(thread_sources_assignment[i][batch])
            end
        end
    end
    Log.info("Done fitting elboargs. Elapsed time: $(toq())")    
    
    # Return add results to dictionary
    results = Dict[]

    for i = 1:n_sources
        entry = catalog[target_sources[i]]
        push!(results, Dict("thing_id"=>entry.thing_id,
                            "objid"=>entry.objid,
                            "ra"=>entry.pos[1],
                            "dec"=>entry.pos[2],
                            "vs"=>model[i].vp[1],
                            "runtime"=>-1))
    end

    results, obj_values
end
