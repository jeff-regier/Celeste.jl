module ParallelRun

using Base.Threads
using Base.Dates

import FITSIO
import JLD
import WCS
using DataStructures

import ..Celeste: detect_sources, BoundingBox, Config
import ..Log
using ..Model
import ..PSF
import ..Coordinates: angular_separation, match_coordinates
import ..MCMC

using ..DeterministicVI
using ..DeterministicVI.ConstraintTransforms: ConstraintBatch, DEFAULT_CHUNK
using ..DeterministicVI.ElboMaximize: ElboConfig, maximize!

include("partition.jl")

"""
Return a Cyclades partitioning or an equal partitioning of the target
sources.
"""
function partition_box(npartitions::Int, target_sources::Vector{Int},
                       neighbor_map::Dict{Int,Vector{Int}}, ea_vec;
                       cyclades_partition=true,
                       batch_size=7000)
    if cyclades_partition
        #return partition_cyclades(npartitions, target_sources,
        #                          neighbor_map,
        #                          batch_size=batch_size)
        #return partition_cyclades_dynamic(target_sources,
        #                                  neighbor_map,
        #                                  batch_size=batch_size)
	return partition_cyclades_dynamic_auto_batchsize(target_sources, neighbor_map, ea_vec)
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
function partition_cyclades_dynamic_auto_batchsize(target_sources::Vector{Int}, neighbor_map::Dict{Int,Vector{Int}}, ea_vec)
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
    cfg_vec = Vector{ElboConfig{DEFAULT_CHUNK,Float64}}(n_sources)

    return ea_vec, vp_vec, cfg_vec, ts_vp
end


"""
Uses multiple threads on one node to fit the Celeste
model over numerous iterations.

catalog - the catalog of light sources
target_sources - light sources to optimize
neighbor_map - light_source id -> neighbor light_source ids

cyclades_partition - use the cyclades algorithm to partition into non conflicting batches for updates.
batch_size - size of a single batch of sources for updates
within_batch_shuffling - whether or not to process sources within a batch randomly
joint_inference_terminate - whether to terminate once sources seem to be stable

Returns:

- Vector of OptimizedSource results
"""
function one_node_joint_infer(catalog, patches, target_sources, neighbor_map,
                              images;
                              cyclades_partition::Bool=true,
                              batch_size::Int=7000,
                              within_batch_shuffling::Bool=true,
                              config=Config())
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

    # Initialize elboargs in parallel
    thread_initialize_sources_assignment::Vector{Vector{Vector{Int64}}} = partition_equally(n_threads, n_sources)

    initialize_elboargs_sources!(config, ea_vec, vp_vec, cfg_vec, thread_initialize_sources_assignment,
                                 catalog, patches, target_sources, neighbor_map, images,
                                 target_source_variational_params)

    #thread_sources_assignment = partition_box(n_threads, target_sources,
    #                                  neighbor_map;
    #                                  cyclades_partition=cyclades_partition,
    #                                  batch_size=batch_size)
    batched_connected_components = partition_box(n_threads, target_sources,
                                                 neighbor_map, ea_vec;
                                                 cyclades_partition=cyclades_partition,
                                                 batch_size=batch_size)

    # Process sources in parallel
    #process_sources!(images, ea_vec, vp_vec, cfg_vec,
    #                 thread_sources_assignment,
    #                 n_iters, within_batch_shuffling)
    niters = config.num_joint_vi_iters
    process_sources_dynamic!(images, ea_vec, vp_vec, cfg_vec,
                             batched_connected_components,
                             niters, within_batch_shuffling)

    # Return add results to vector
    results = OptimizedSource[]

    for i = 1:n_sources
        entry = catalog[target_sources[i]]
        is_sky_bad = bad_sky(entry, images)
        result = OptimizedSource(entry.pos[1], entry.pos[2], vp_vec[i][1],
                                 is_sky_bad)
        push!(results, result)
    end

    show_pixels_processed()

    results
end

function initialize_elboargs_sources!(config::Config, ea_vec, vp_vec, cfg_vec,
                                      thread_initialize_sources_assignment,
                                      catalog, patches, target_sources, neighbor_map, images,
                                      target_source_variational_params;
                                      termination_callback=nothing)
    Threads.@threads for i in 1:nthreads()
        try
            for batch in 1:length(thread_initialize_sources_assignment[i])
                for source_index in thread_initialize_sources_assignment[i][batch]
                    init_elboargs(config, source_index, catalog, patches, target_sources,
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
function init_elboargs(config::Config,
                       ts::Int,
                       catalog::Vector{CatalogEntry},
                       patches::Matrix{ImagePatch},
                       target_sources::Vector{Int},
                       neighbor_map::Dict{Int,Vector{Int}},
                       images::Vector{<:Image},
                       ea_vec::Vector{ElboArgs},
                       vp_vec::Vector{VariationalParams{Float64}},
                       cfg_vec::Vector{ElboConfig{DEFAULT_CHUNK,Float64}},
                       ts_vp::Dict{Int64,Array{Float64}};
                       termination_callback=nothing)
    try
        entry_id = target_sources[ts]
        entry = catalog[entry_id]
        neighbor_ids = neighbor_map[entry_id]
        neighbors = catalog[neighbor_ids]
        cat_local = vcat([entry], neighbors)
        ids_local = vcat([entry_id], neighbor_ids)

        # Limit patches to just the active source and its neighbors.
        patches = patches[ids_local, :]

        # Load vp with shared target source params, and also vp
        # that doesn't share target source params
        vp = Vector{Float64}[haskey(ts_vp, x) ?
                             ts_vp[x] :
                             catalog_init_source(catalog[x])
                             for x in ids_local]
        ea = ElboArgs(images, patches, [1])

        ea_vec[ts] = ea
        vp_vec[ts] = vp
        cfg_vec[ts] = ElboConfig(ea, vp;
                termination_callback=termination_callback)
    catch exc
        if is_production_run || nthreads() > 1
            Log.exception(exc)
        else
            rethrow()
        end
    end
end


function process_sources!(images::Vector{<:Model.Image},
                          ea_vec::Vector{ElboArgs},
                          vp_vec::Vector{VariationalParams{Float64}},
                          cfg_vec::Vector{ElboConfig{DEFAULT_CHUNK,Float64}},
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

function process_sources_dynamic!(images::Vector{<:Model.Image},
                                  ea_vec::Vector{ElboArgs},
                                  vp_vec::Vector{VariationalParams{Float64}},
                                  cfg_vec::Vector{ElboConfig{DEFAULT_CHUNK,Float64}},
                                  thread_sources_assignment::Vector{Vector{Vector{Int64}}},
                                  n_iters::Int,
                                  within_batch_shuffling::Bool)
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
                    end
                    process_sources_elapsed_times[i::Int] = toq()
                catch exc
                    Log.exception(exc)
                    rethrow()
                end
            end
            Log.info("Batch $(batch) - $(process_sources_elapsed_times)")
	    avg_thread_idle_time = mean(maximum(process_sources_elapsed_times) - process_sources_elapsed_times)
	    maximum_thread_time = maximum(process_sources_elapsed_times)
            idle_percent = 100.0 * (avg_thread_idle_time / maximum_thread_time)
            Log.info("Batch $(batch) avg threads idle: $(round(Int, idle_percent))% ($(avg_thread_idle_time) / $(maximum_thread_time))")
	    total_idle_time += sum(maximum(process_sources_elapsed_times) - process_sources_elapsed_times)
            total_sum_of_thread_times += sum(process_sources_elapsed_times)
        end
    end
    Log.info("Total idle time: $(round(Int, total_idle_time)), Total sum of threads times: $(round(Int, total_sum_of_thread_times))")
end

# Process partition of sources. Multiple threads call this function in parallel.
function process_sources_kernel!(ea_vec::Vector{ElboArgs},
                                 vp_vec::Vector{VariationalParams{Float64}},
                                 cfg_vec::Vector{ElboConfig{DEFAULT_CHUNK,Float64}},
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
    Log.info("$(Time(now())): (active,inactive) pixels processed: \($n_active,$n_inactive\)")
end


# In production mode, rather the development mode, always catch exceptions
const is_production_run = haskey(ENV, "CELESTE_PROD") && ENV["CELESTE_PROD"] != ""

# ------
# optimization

# optimization result container
struct OptimizedSource
    init_ra::Float64
    init_dec::Float64
    vs::Vector{Float64}
    is_sky_bad::Bool
end


# The sky intensity does not appear to match the true background intensity
# for some light sources. This function flags light sources whose infered
# brightness may be inaccurate because the background intensity estimates
# are off.
function bad_sky(ce, images)
    # The 'i' band is a pretty bright one, so I used it here.
    img_index = findfirst(img -> img.b == 4, images)
    if img_index < 0
        return false
    end

    img = images[img_index]

    # Get sky at location of the object
    pixel_center = WCS.world_to_pix(img.wcs, ce.pos)
    h = max(1, min(round(Int, pixel_center[1]), size(img.pixels, 1)))
    w = max(1, min(round(Int, pixel_center[2]), size(img.pixels, 2)))

    claimed_sky = img.sky[h, w] * img.nelec_per_nmgy[h]

    # Determine background in a 50-pixel radius box around the object
    box = Model.box_around_point(img.wcs, ce.pos, 50.0)
    h_range, w_range = Model.clamp_box(box, (img.H, img.W))
    observed_sky = median(filter(!isnan, img.pixels[h_range, w_range]))

    # A 5 photon-per-pixel disparity can really add up if the light sources
    # covers a lot of pixels.
    return (claimed_sky + 5) < observed_sky
end


"""
Optimize the `s`th element in `catalog`.
Used only for one_node_single_infer, not one_node_joint_infer.
"""
function process_source(config::Config,
                        s::Int,
                        catalog::Vector{CatalogEntry},
                        patches::Matrix{ImagePatch},
                        neighbor_ids::Vector{Int},
                        images::Vector{<:Image})
    neighbors = catalog[neighbor_ids]
    if length(neighbors) > 100
        msg = string("object at RA, Dec = $(entry.pos) has an excessive",
                     "number ($(length(neighbors))) of neighbors")
        Log.warn(msg)
    end

    entry = catalog[s]
    cat_local = vcat([entry], neighbors)

    # Limit patches to just the active source and its neighbors.
    idxs = vcat([s], neighbor_ids)
    patches = patches[idxs, :]

    vp = DeterministicVI.init_sources([1], cat_local)
    ea = DeterministicVI.ElboArgs(images, patches, [1])

    tic()
    f_evals, max_f, max_x, nm_result = DeterministicVI.ElboMaximize.maximize!(ea, vp)
    Log.info("#$(s) at ($(entry.pos[1]), $(entry.pos[2])): $(toq()) secs")

    vs_opt = vp[1]
    is_sky_bad = bad_sky(entry, images)
    return OptimizedSource(entry.pos[1], entry.pos[2], vs_opt, is_sky_bad)
end


"""
Run MCMC to process a source.  Returns
"""
function process_source_mcmc(config::Config,
                             s::Int,
                             catalog::Vector{CatalogEntry},
                             patches::Matrix{ImagePatch},
                             neighbor_ids::Vector{Int},
                             images::Vector{<:Image};
                             use_ais::Bool=true)
    # subselect source, select active source and neighbor set
    entry = catalog[s]
    neighbors = catalog[neighbor_ids]
    if length(neighbors) > 100
        msg = string("objid $(entry.objid) [ra: $(entry.pos)] has an excessive",
                     "number ($(length(neighbors))) of neighbors")
        Log.warn(msg)
    end

    # Limit patches to just the active source and its neighbors.
    idxs = vcat([s], neighbor_ids)
    patches = patches[idxs, :]

    # create smaller images for the MCMC sampler to use
    patch_images = [MCMC.patch_to_image(patches[1, i], images[i])
                    for i in 1:length(images)]

    # render a background image on the active source (first in list)
    background_images = [MCMC.render_patch_nmgy(patch_images[i], patches[1, i], neighbors)
                         for i in 1:length(patch_images)]

    # run mcmc sampler on this image/patch/background initialized at entry
    if use_ais
        mcmc_results = MCMC.run_ais(entry, patch_images, patches, background_images;
            num_temperatures=config.num_ais_temperatures,
            num_samples=config.num_ais_samples)
    else
        mcmc_results = MCMC.run_mcmc(entry, patch_images, patches, background_images)
    end

    # summary
    return mcmc_results
end


function one_node_single_infer(catalog::Vector{CatalogEntry},
                               patches::Matrix{ImagePatch},
                               target_sources::Vector{Int},
                               neighbor_map::Dict{Int,Vector{Int}},
                               images::Vector{<:Image};
                               config=Config(),
                               do_vi=true)
    curr_source = 1
    last_source = length(target_sources)
    sources_lock = SpinLock()
    results_lock = SpinLock()
    if do_vi
        results = OptimizedSource[]
    else
        results = []
    end

    # iterate over sources
    function process_sources()
        tid = threadid()

        while true
            lock(sources_lock)
            ts = curr_source
            curr_source += 1
            unlock(sources_lock)
            if ts > last_source
                break
            end

            s = target_sources[ts]
            try
                if do_vi
                    result = process_source(config, s, catalog, patches,
                                            neighbor_map[s], images)
                else
                    result = process_source_mcmc(config, s, catalog, patches,
                                                 neighbor_map[s], images)
                end

                lock(results_lock)
                push!(results, result)
                unlock(results_lock)
            catch ex
                if is_production_run || nthreads() > 1
                    Log.exception(ex)
                else
                    rethrow(ex)
                end
            end
        end
    end

    if nthreads() == 1
        process_sources()
    else
        ccall(:jl_threading_run, Void, (Any,), Core.svec(process_sources))
        Log.LEVEL[] >= Log.INFO && ccall(:jl_threading_profile, Void, ())
    end

    return results
end


function _infer_box(images::Vector{<:Image}, catalog::Vector{CatalogEntry},
                    patches::Matrix{ImagePatch}, box::BoundingBox;
                    method=:joint_vi, config=Config())
    # Get indices of entries in the RA/Dec range of interest.
    # (Some images can have regions that are outside the box, so not
    # all sources are necessarily in the box.)
    entry_in_range = entry->((box.ramin < entry.pos[1] < box.ramax) &&
                             (box.decmin < entry.pos[2] < box.decmax))
    target_ids = find(entry_in_range, catalog)

    # find objects with patches overlapping
    neighbor_map = Dict(id=>Model.find_neighbors(patches, id)
                        for id in target_ids)

    if method == :joint_vi
        results = one_node_joint_infer(catalog, patches, target_ids,
                                       neighbor_map, images; config=config)
    elseif method == :single_vi
        results = one_node_single_infer(catalog, patches, target_ids,
                                        neighbor_map, images; do_vi=true,
                                        config=config)
    elseif method == :mcmc
        results = one_node_single_infer(catalog, patches, target_ids,
                                        neighbor_map, images; do_vi=false,
                                        config=config)
    else
        error("unknown method: $method")
    end
    return results
end


"""
    infer_box(images, box; options...)

Detect objects in images and run inference on all sources
within the bounding box. If `catalog` is given, run inference on objects in
`catalog` that fall within the bounding box.

Objects outside the bounding box (either detected or given)
may be used as "neighbors".
"""
function infer_box(images::Vector{<:Image}, box::BoundingBox;
                   catalog::Union{Void, Vector{CatalogEntry}}=nothing,
                   method=:joint_vi, config=Config())
    tic()
    Log.info("processing box $(box.ramin), $(box.ramax), $(box.decmin), ",
             "$(box.decmax) with $(nthreads()) threads")

    if catalog === nothing
        cat, patches = detect_sources(images)
    else
        cat = catalog
        patches = Model.get_sky_patches(images, cat)
    end

    results = _infer_box(images, cat, patches, box;
                         method=method, config=config)

    Log.info("infer_box for $box took $(toq()) seconds")

    return results
end

end
