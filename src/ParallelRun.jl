module ParallelRun

import FITSIO
import JLD

import ..Log
using ..Model
import ..SDSSIO
import ..Infer
import ..SDSSIO: RunCamcolField
import ..PSF

using ..DeterministicVI


#set this to false to use source-division parallelism
const SKY_DIVISION_PARALLELISM=true

const TILE_WIDTH = 20
const MIN_FLUX = 2.0

# Use distributed parallelism (with Dtree)
if haskey(ENV, "USE_DTREE") && ENV["USE_DTREE"] != ""
    const Distributed = true
    using Dtree
else
    const Distributed = false
    const dt_nodeid = 1
    const dt_nnodes = 1
    DtreeScheduler(n, f) = ()
    initwork(dt) = 0, (1, 0)
    getwork(dt) = 0, (1, 0)
    runtree(dt) = 0
    cpu_pause() = ()
end

# Use threads (on the loop over sources)
const Threaded = true
if Threaded && VERSION > v"0.5.0-dev"
    using Base.Threads
else
    # Pre-Julia 0.5 there are no threads
    nthreads() = 1
    threadid() = 1
    macro threads(x)
        x
    end
    SpinLock() = 1
    lock(l) = ()
    unlock(l) = ()
end


# A workitem is of this ra / dec size
const wira = 0.025
const widec = 0.025


"""
Timing information.
"""
type InferTiming
    query_fids::Float64
    num_infers::Int64
    read_photoobj::Float64
    read_img::Float64
    fit_psf::Float64
    opt_srcs::Float64
    num_srcs::Int64
    write_results::Float64
    wait_done::Float64

    InferTiming() = new(0.0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)
end


function add_timing!(i::InferTiming, j::InferTiming)
    i.query_fids = i.query_fids + j.query_fids
    i.num_infers = i.num_infers + j.num_infers
    i.read_photoobj = i.read_photoobj + j.read_photoobj
    i.read_img = i.read_img + j.read_img
    i.fit_psf = i.fit_psf + j.fit_psf
    i.opt_srcs = i.opt_srcs + j.opt_srcs
    i.num_srcs = i.num_srcs + j.num_srcs
    i.write_results = i.write_results + j.write_results
    i.wait_done = i.wait_done + j.wait_done
end


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


@inline nputs(nid, s) = ccall(:puts, Cint, (Ptr{Int8},), string("[$nid] ", s))
@inline phalse(b) = b[] = false


include("source_division_inference.jl")


"""
Given a BoundingBox that is to be divided into `nra` x `ndec` subareas,
return the `i`th subarea. `i` is a linear index between 1 and
`nra * ndec`.

This function assumes a cartesian (rather than spherical) coordinate system!
"""
function divide_skyarea(box, nra, ndec, i)
    global wira, widec
    ix, iy = ind2sub((nra, ndec), i)

    return (box.ramin + (ix - 1) * wira,
            min(box.ramin + ix * wira, box.ramax),
            box.decmin + (iy - 1) * widec,
            min(box.decmin + iy * widec, box.decmax))
end


function time_puts(elapsedtime, bytes, gctime, allocs)
    s = @sprintf("%10.6f seconds", elapsedtime/1e9)
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
    nputs(dt_nodeid, s)
end


"""
Divide `N` into `np` parts as evenly as possible, returning `my` part as
a (first, last) tuple.
"""
function divparts(N, np, my)
    len, rem = divrem(N, np)
    if len == 0
        if my > rem
            return 1, 0
        end
        len, rem = 1, 0
    end
    # compute my part
    f = 1 + ((my-1) * len)
    l = f + len - 1
    # distribute remaining evenly
    if rem > 0
        if my <= rem
            f = f + (my-1)
            l = l + my
        else
            f = f + rem
            l = l + rem
        end
    end
    return f, l
end


"""
Divide the given ra, dec range into sky areas of `wira`x`widec` and
use Dtree to distribute these sky areas to nodes. Within each node
use `one_node_infer()` to fit the Celeste model to sources in each sky area.
"""
function divide_sky_and_infer(
                box::BoundingBox,
                stagedir::String;
                timing=InferTiming(),
                outdir=".")
    if dt_nodeid == 1
        nputs(dt_nodeid, "running on $dt_nnodes nodes")
    end

    # how many `wira` X `widec` sky areas (work items)?
    global wira, widec
    nra = ceil(Int64, (box.ramax - box.ramin) / wira)
    ndec = ceil(Int64, (box.decmax - box.decmin) / widec)

    num_work_items = nra * ndec
    each = ceil(Int64, num_work_items / dt_nnodes)

    if dt_nodeid == 1
        nputs(dt_nodeid, "work item dimensions: $wira X $widec")
        nputs(dt_nodeid, "$num_work_items work items, ~$each per node")
    end

    # create Dtree and get the initial allocation
    dt, is_parent = DtreeScheduler(num_work_items, 0.4)
    ni, (ci, li) = initwork(dt)
    rundt = Ref(runtree(dt))
    @inline function rundtree(again)
        if again[]
            again[] = runtree(dt)
            cpu_pause()
        end
        again[]
    end

    # work item processing loop
    nputs(dt_nodeid, "initially $ni work items ($ci to $li)")
    itimes = InferTiming()
    while ni > 0
        li == 0 && break
        if ci > li
            nputs(dt_nodeid, "consumed allocation (last was $li)")
            ni, (ci, li) = getwork(dt)
            nputs(dt_nodeid, "got $ni work items ($ci to $li)")
            continue
        end
        item = ci
        ci = ci + 1

        # map item to subarea
        iramin, iramax, idecmin, idecmax = divide_skyarea(box, nra, ndec, item)

        # Get vector of (run, camcol, field) triplets overlapping this patch
        tic()
        box = BoundingBox(iramin, iramax, idecmin, idecmax)
        rcfs = get_overlapping_fields(box, stagedir)
        itimes.query_fids = toq()

        # run inference for this subarea
        results, obj_value = one_node_infer(rcfs, stagedir;
                                            box=BoundingBox(iramin, iramax, idecmin, idecmax),
                                            reserve_thread=rundt,
                                            thread_fun=rundtree,
                                            timing=itimes)
        tic()
        save_results(outdir, iramin, iramax, idecmin, idecmax, results)
        itimes.write_results = toq()

        timing.num_infers = timing.num_infers+1
        add_timing!(timing, itimes)
        rundtree(rundt)
    end
    nputs(dt_nodeid, "out of work")
    tic()
    while rundt[]
        rundtree(rundt)
    end
    finalize(dt)
    timing.wait_done = toq()
end


function load_images(rcfs, stagedir)
    images = TiledImage[]
    image_names = String[]
    image_count = 0

    for i in 1:length(rcfs)
        Log.info("reading field $(rcfs[i])")
        rcf = rcfs[i]
        field_images = SDSSIO.load_field_images(rcf, stagedir)
        for b=1:length(field_images)
            image_count += 1
            push!(image_names,
                "$image_count run=$(rcf.run) camcol=$(rcf.camcol) field=$(rcf.field) b=$b")
            tiled_image = TiledImage(field_images[b])
            push!(images, tiled_image)
        end
    end
    gc()

    Log.debug("Image names:")
    Log.debug(string(image_names))

    images
end

function union_find(i, components_tree)
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
        target = union_find(i-source_start+1, components_tree)
        for neighbor in neighbor_map[index_to_source[i-source_start+1]]
            if haskey(source_to_index, neighbor)
                neighbor_idx = source_to_index[neighbor]                
                component_source = union_find(neighbor_idx, components_tree)
                components_tree[component_source] = target
            end
            
        end
    end

    # Write to components dictionary.
    for i = source_start:source_end
        index = union_find(i-source_start+1, components_tree)
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
    shuffle!(sources)

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
    n_sources_per_thread = convert(Int64, floor(n_sources / nprocthreads))
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

- rcfs: Array of run, camcol, field triplets that the source occurs in.
- box: a bounding box specifying a region of sky

Returns:

- Dictionary of results, keyed by SDSS thing_id.
"""
function one_node_infer_multi_iter(rcfs::Vector{RunCamcolField},
                                   stagedir::String;
                                   cyclades_partition=true,
                                   n_iters=10,
                                   objid="",
                                   box=BoundingBox(-1000., 1000., -1000., 1000.),
                                   primary_initialization=true,
                                   reserve_thread=Ref(false),
                                   thread_fun=phalse,
                                   timing=InferTiming())
    nprocthreads = nthreads()
    if reserve_thread[]
        nprocthreads = nprocthreads-1
    end
    Log.info("Running with $(nprocthreads) threads")
    
    # Read all primary objects in these fields.
    tic()
    duplicate_policy = primary_initialization ? :primary : :first
    catalog = SDSSIO.read_photoobj_files(rcfs, stagedir,
                        duplicate_policy=duplicate_policy)
    timing.read_photoobj = toq()
    Log.info("$(length(catalog)) primary sources")

    reserve_thread[] && thread_fun(reserve_thread)

    # Filter out low-flux objects in the catalog.
    catalog = filter(entry->(maximum(entry.star_fluxes) >= MIN_FLUX), catalog)
    Log.info("$(length(catalog)) primary sources after MIN_FLUX cut")

    # Filter any object not specified, if an objid is specified
    if objid != ""
        Log.info(catalog[1].objid)
        catalog = filter(entry->(entry.objid == objid), catalog)
    end

    # Get indicies of entries in the  RA/Dec range of interest.
    entry_in_range = entry->((box.ramin < entry.pos[1] < box.ramax) &&
                             (box.decmin < entry.pos[2] < box.decmax))
    target_sources = find(entry_in_range, catalog)

    nputs(dt_nodeid, string("processing $(length(target_sources)) sources in ",
          "$(box.ramin), $(box.ramax), $(box.decmin), $(box.decmax)"))

    # If there are no objects of interest, return early.
    if length(target_sources) == 0
        return Dict{Int, Dict}()
    end

    reserve_thread[] && thread_fun(reserve_thread)

    # Read in images for all (run, camcol, field).
    tic()

    images = load_images(rcfs, stagedir)
    timing.read_img = toq()

    reserve_thread[] && thread_fun(reserve_thread)

    Log.info("Finding neighbors")
    tic()
    neighbor_map = Infer.find_neighbors(target_sources, catalog, images)
    Log.info("Neighbors found in $(toq()) seconds")

    reserve_thread[] && thread_fun(reserve_thread)

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
    # TODO max: parallelize?
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

""" 
Use mulitple threads on one node to 
fit the Celeste model to sources in a given bounding box.

- rcfs: Array of run, camcol, field triplets that the source occurs in.
- box: a bounding box specifying a region of sky

Returns:

- Dictionary of results, keyed by SDSS thing_id.
"""
function one_node_infer(
               rcfs::Vector{RunCamcolField},
               stagedir::String;
               objid="",
               box=BoundingBox(-1000., 1000., -1000., 1000.),
               primary_initialization=true,
               reserve_thread=Ref(false),
               thread_fun=phalse,
               timing=InferTiming())
    nprocthreads = nthreads()
    if reserve_thread[]
        nprocthreads = nprocthreads-1
    end
    Log.info("Running with $(nprocthreads) threads")

    # Read all primary objects in these fields.
    tic()
    duplicate_policy = primary_initialization ? :primary : :first
    catalog = SDSSIO.read_photoobj_files(rcfs, stagedir,
                        duplicate_policy=duplicate_policy)
    timing.read_photoobj = toq()
    Log.info("$(length(catalog)) primary sources")

    reserve_thread[] && thread_fun(reserve_thread)

    # Filter out low-flux objects in the catalog.
    catalog = filter(entry->(maximum(entry.star_fluxes) >= MIN_FLUX), catalog)
    Log.info("$(length(catalog)) primary sources after MIN_FLUX cut")

    # Filter any object not specified, if an objid is specified
    if objid != ""
        Log.info(catalog[1].objid)
        catalog = filter(entry->(entry.objid == objid), catalog)
    end

    # Get indicies of entries in the  RA/Dec range of interest.
    entry_in_range = entry->((box.ramin < entry.pos[1] < box.ramax) &&
                             (box.decmin < entry.pos[2] < box.decmax))
    target_sources = find(entry_in_range, catalog)

    nputs(dt_nodeid, string("processing $(length(target_sources)) sources in ",
          "$(box.ramin), $(box.ramax), $(box.decmin), $(box.decmax)"))

    # If there are no objects of interest, return early.
    if length(target_sources) == 0
        return Dict{Int, Dict}()
    end

    reserve_thread[] && thread_fun(reserve_thread)

    # Read in images for all (run, camcol, field).
    tic()

    images = load_images(rcfs, stagedir)
    timing.read_img = toq()

    reserve_thread[] && thread_fun(reserve_thread)

    Log.info("finding neighbors")
    tic()
    neighbor_map = Infer.find_neighbors(target_sources, catalog, images)
    Log.info("neighbors found in $(toq()) seconds")

    reserve_thread[] && thread_fun(reserve_thread)

    # iterate over sources
    obj_values = Array{Float64}(length(target_sources))
    curr_source = 1
    last_source = length(target_sources)
    sources_lock = SpinLock()
    results = Dict[]
    results_lock = SpinLock()
    function process_sources()
        tid = threadid()

        if reserve_thread[] && tid == 1
            while reserve_thread[]
                thread_fun(reserve_thread)
                cpu_pause()
            end
        else
            while true
                lock(sources_lock)
                ts = curr_source
                curr_source += 1
                unlock(sources_lock)
                if ts > last_source
                    break
                end
#                try
                    s = target_sources[ts]
                    entry = catalog[s]
                    nputs(dt_nodeid, "processing source $s: objid = $(entry.objid)")

                    t0 = time()
                    # TODO: subset images to images_local too.
                    vs_opt, obj_value = Infer.infer_source(images,
                                                           catalog[neighbor_map[ts]],
                                                           entry)
                    runtime = time() - t0
                    obj_values[ts] = obj_value
#                catch ex
#                    Log.error(ex)
#                end

                lock(results_lock)
                println(vs_opt)
                push!(results, Dict(
                    "thing_id"=>entry.thing_id,
                    "objid"=>entry.objid,
                    "ra"=>entry.pos[1],
                    "dec"=>entry.pos[2],
                    "vs"=>vs_opt,
                    "runtime"=>runtime))
                unlock(results_lock)
            end
        end
    end

    tic()
    ccall(:jl_threading_run, Void, (Any,), Core.svec(process_sources))
    ccall(:jl_threading_profile, Void, ())
    timing.opt_srcs = toq()
    timing.num_srcs = length(target_sources)
    
    results, obj_values
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


"""
Like `get_overlapping_fields`, but return a Vector of
(run, camcol, field) triplets.
"""
function get_overlapping_fields(query::BoundingBox, stagedir::String)
    fes = get_overlapping_field_extents(query, stagedir)
    [fe[1] for fe in fes]
end


"""
called from main entry point for inference for one field
(used for accuracy assessment, infer-box is the primary inference
entry point)
"""
function infer_field(rcf::RunCamcolField,
                     stagedir::String,
                     outdir::String;
                     objid="")
    results, obj_value = one_node_infer([rcf,], stagedir; objid=objid, primary_initialization=false)
    fname = if objid == ""
        @sprintf "%s/celeste-%06d-%d-%04d.jld" outdir rcf.run rcf.camcol rcf.field
    else
        @sprintf "%s/celeste-objid-%s.jld" outdir objid
    end
    JLD.save(fname, "results", results)
    Log.debug("infer_field finished successfully")
end


"""
Save provided results to a JLD file.
"""
function save_results(outdir, ramin, ramax, decmin, decmax, results)
    fname = @sprintf("%s/celeste-%.4f-%.4f-%.4f-%.4f.jld",
                     outdir, ramin, ramax, decmin, decmax)
    JLD.save(fname, "results", results)
end

save_results(outdir, box, results) =
    save_results(outdir, box.ramin, box.ramax, box.decmin, box.decmax, results)


"""
called from main entry point.
"""
function infer_box(box::BoundingBox, stagedir::String, outdir::String)
    # Base.@time hack for distributed environment
    gc_stats = ()
    gc_diff_stats = ()
    elapsed_time = 0.0
    gc_stats = Base.gc_num()
    elapsed_time = time_ns()

    times = InferTiming()
    if !SKY_DIVISION_PARALLELISM
        Log.debug("source division parallelism")
        divide_sources_and_infer(box, stagedir; timing=times, outdir=outdir)
    elseif dt_nnodes > 1
        Log.debug("sky division parallelism")
        divide_sky_and_infer(box, stagedir; timing=times, outdir=outdir)
    else
        Log.debug("multithreaded parallelism only")
        tic()
        # Get vector of (run, camcol, field) triplets overlapping this patch
        rcfs = get_overlapping_fields(box, stagedir)
        times.query_fids = toq()

        results, obj_value = one_node_infer(rcfs, stagedir; box=box, timing=times)

        tic()
        save_results(outdir, box, results)
        times.write_results = toq()
    end

    # Base.@time hack for distributed environment
    elapsed_time = time_ns() - elapsed_time
    gc_diff_stats = Base.GC_Diff(Base.gc_num(), gc_stats)
    time_puts(elapsed_time, gc_diff_stats.allocd, gc_diff_stats.total_time,
              Base.gc_alloc_count(gc_diff_stats))

    times.num_srcs = max(1, times.num_srcs)
    nputs(dt_nodeid, "timing: query_fids=$(times.query_fids)")
    nputs(dt_nodeid, "timing: num_infers=$(times.num_infers)")
    nputs(dt_nodeid, "timing: read_photoobj=$(times.read_photoobj)")
    nputs(dt_nodeid, "timing: read_img=$(times.read_img)")
    nputs(dt_nodeid, "timing: fit_psf=$(times.fit_psf)")
    nputs(dt_nodeid, "timing: opt_srcs=$(times.opt_srcs)")
    nputs(dt_nodeid, "timing: num_srcs=$(times.num_srcs)")
    nputs(dt_nodeid, "timing: average opt_srcs=$(times.opt_srcs/times.num_srcs)")
    nputs(dt_nodeid, "timing: write_results=$(times.write_results)")
    nputs(dt_nodeid, "timing: wait_done=$(times.wait_done)")
end

end
