import JLD
import ..Infer
import ..ParallelRun: infer_init, one_node_infer, BoundingBox,
    one_node_joint_infer, one_node_single_infer
import ..SensitiveFloats: SensitiveFloat

"""
load_ea_from_source
Helper function to load elbo args for a particular source
"""
function load_ea_from_source(target_source, target_sources, catalog, images, all_vps; use_fft=false)
    # Get neighbors of the source from which to load elbo args.
    neighbor_map = Infer.find_neighbors([target_source], catalog, images)

    # Create a dictionary to the optimized parameters.
    target_source_variational_params = Dict{Int64, Array{Float64}}()
    for (indx, cur_source) in enumerate(target_sources)
        target_source_variational_params[cur_source] = all_vps[indx]
    end

    # Load neighbors, patches and variational parameters
    neighbors = catalog[neighbor_map[1]]
    entry = catalog[target_source]
    cat_local = vcat([entry], neighbors)
    patches = Infer.get_sky_patches(images, cat_local)
    Infer.load_active_pixels!(images, patches)
    ids_local = vcat([target_source], neighbor_map[1])
    vp = [haskey(target_source_variational_params, x) ?
          target_source_variational_params[x] :
          DeterministicVI.catalog_init_source(catalog[x]) for x in ids_local]

    # Create elbo args
    fsm_mat = 0
    if use_fft
        ea, fsm_mat = DeterministicVIImagePSF.initialize_fft_elbo_parameters(images,
                                                                             vp,
                                                                             patches,
                                                                             [1],
                                                                             use_raw_psf=false)
    else
        ea = ElboArgs(images, vp, patches, [1])
    end
    ea, fsm_mat
end

"""
compute_unconstrained_gradient
"""
function compute_unconstrained_gradient(target_source, target_sources, catalog, images, all_vps; use_fft=false)
    # Load ea
    (ea, fsm) = load_ea_from_source(target_source, target_sources, catalog, images,
                                    all_vps, use_fft=use_fft)

    # Evaluate in constrained space and then unconstrain. (Taken from maximize_f)
    last_sf::SensitiveFloat{Float64} = SensitiveFloats.SensitiveFloat{Float64}(length(UnconstrainedParams), 1, true, true)
    transform = DeterministicVI.get_mp_transform(ea.vp, ea.active_sources)
    elbo = use_fft ? get_fft_elbo_function(ea, fsm) : DeterministicVI.elbo
    f_res = elbo(ea)
    Transform.transform_sensitive_float!(transform, last_sf, f_res, ea.vp, ea.active_sources)
    last_sf.d
end

"""
unconstrained_gradient_near_zero
"""
function unconstrained_gradient_near_zero(target_sources, catalog, images, all_vps; use_fft=false)

    # Compute gradient per source
    for (cur_source_indx, source) in enumerate(target_sources)
        gradient = compute_unconstrained_gradient(source, target_sources, catalog, images,all_vps; use_fft=use_fft)
        if !isapprox(gradient, zeros(length(gradient)), atol=1)
            return false
        end
    end
    return true
end

"""
compare_vp_params
Helper to check whether two sets of variational parameters from the
result of joint inference are the same.

Return true if vp params are the same, false otherwise
"""
function compare_vp_params(r1, r2)

    # Create a map from thingid -> vp for r1
    r1_vp = Dict{Int64, Vector{Float64}}()
    for r1_result in r1
        r1_vp[r1_result.thingid] = r1_result.vs
    end

    # Check the existence and equivalence of each source's vp in r2
    for r2_result in r2
        if !haskey(r1_vp, r2_result.thingid) || r1_vp[r2_result.thingid] != r2_result.vs
            if r1_vp[r2_result.thingid] != r2_result.vs
                println("compare_vp_params: Mismatch - $(r1_vp[r2_result.thingid]) vs $(r2_result.vs)")
            end
            return false
        end
    end

    return length(r1) == length(r2)
end

"""
compute_obj_value
computes obj value given set of results from one_node_infer and one_node_joint_infer
"""
function compute_obj_value(results,
                           rcfs::Vector{RunCamcolField},
                           stagedir::String;
                           box=BoundingBox(-1000.,1000.,-1000.,1000.),
                           primary_initialization=true)
    catalog, target_sources = infer_init(rcfs, stagedir; box=box, primary_initialization=primary_initialization)
    images = SDSSIO.load_field_images(rcfs, stagedir)

    vp = [r.vs for r in results]

    # it may be better to pass `patches` as an argument to `compute_obj_value`.
    patches = Infer.get_sky_patches(images, catalog[target_sources])

    # if we don't call `load_active_pixels!`, these patches will be different
    # than the patches we used for optimizing---and then the objective
    # function could also be slightly different
    Infer.load_active_pixels!(images, patches)

    # this works since we're just generating patches for active_sources.
    # if instead you pass patches for all sources, then instead we'd used
    # active_sources = target_sources
    active_sources = collect(1:length(target_sources))

    ea = ElboArgs(images, vp, patches, active_sources,
                  calculate_gradient=false, calculate_hessian=false)
    DeterministicVI.elbo(ea).v[]
end

function test_fft_on_one_source_matches_single()
    # This bounding box has 1 source.
    box = BoundingBox(164.37, 164.38, 39.10, 39.13)
    field_triplets = [RunCamcolField(3900, 6, 269),]

    # Joint infer. Don't use default optim params since single infer does not.
    infer_joint(ctni...) = one_node_joint_infer(ctni...;
                                                n_iters=1,
                                                use_fft=true,
                                                use_default_optim_params=false)
    result_infer_joint = one_node_infer(field_triplets, datadir;
                                        infer_callback=infer_joint,
                                        box=box)

    # Single infer fft.
    infer_source_callback = DeterministicVIImagePSF.infer_source_fft
    infer_single(ctni...) = one_node_single_infer(ctni...;
                                                  infer_source_callback=infer_source_callback)
    result_infer_single = one_node_infer(field_triplets, datadir;
                                         infer_callback=infer_single,
                                         box=box)

    # Make sure that parameters are exactly the same
    @test compare_vp_params(result_infer_joint, result_infer_single)
end

"""
load_stripe_82_data
"""
function load_stripe_82_data()
    rcf = RunCamcolField(4263, 5, 119)
    cd(datadir)
    run(`make RUN=$(rcf.run) CAMCOL=$(rcf.camcol) FIELD=$(rcf.field)`)
    cd(wd)
    catalog, target_sources = infer_init([rcf], datadir, primary_initialization=false)
    images = SDSSIO.load_field_images([rcf], datadir)
    ([rcf], datadir, target_sources, catalog, images)
end

"""
test_improve_stripe_82_obj_value
"""
function test_improve_stripe_82_obj_value(; use_fft=false)
    println("Testing that joint_infer improves score on stripe 82...")
    (rcfs, datadir, target_sources, catalog, images) = load_stripe_82_data()

    # Single inference obj value
    infer_source_callback = use_fft ?
        DeterministicVIImagePSF.infer_source_fft :
        DeterministicVI.infer_source
    infer_single(ctni...) = one_node_single_infer(ctni...;
                                                  infer_source_callback=infer_source_callback)
    result_single = one_node_infer(rcfs, datadir;
                                   infer_callback=infer_single,
                                   primary_initialization=false)
    score_single = compute_obj_value(result_single, rcfs, datadir;
                                     primary_initialization=false)

    # Joint inference obj value
    infer_multi(ctni...) = one_node_joint_infer(ctni...;
                                                n_iters=30,
                                                within_batch_shuffling=true,
                                                use_fft=use_fft)
    result_multi = one_node_infer(rcfs, datadir;
                                  infer_callback=infer_multi,
                                  primary_initialization=false)
    score_multi = compute_obj_value(result_multi, rcfs, datadir;
                                    primary_initialization=false)

    println("Score single: $(score_single)")
    println("Score multi: $(score_multi)")

    @test score_multi > score_single
end

"""
test_gradient_is_near_zero_on_stripe_82
TODO(max) - This doesn't pass right now since some light sources are not close to zero.
"""
function test_gradient_is_near_zero_on_stripe_82(; use_fft=false)
    (rcfs, datadir, target_sources, catalog, images) = load_stripe_82_data()

    # Make sure joint infer with few iterations does not pass the gradient near zero check
    joint_few(cnti...) = one_node_joint_infer(cnti...; use_fft=use_fft)
    results_few = one_node_infer(rcfs, datadir; infer_callback=joint_few, primary_initialization=false)
    @test !unconstrained_gradient_near_zero(target_sources, catalog, images, [x.vs for x in results_few])

    # Make sure joint infer with many iterations passes the gradient near zero check
    joint_many(cnti...) = one_node_joint_infer(cnti...; use_fft=use_fft, n_iters=10)
    results_many = one_node_infer(rcfs, datadir; infer_callback=joint_many, primary_initialization=false)
    @test unconstrained_gradient_near_zero(target_sources, catalog, images, [x.vs for x in results_many])
end

"""
test_gradient_is_near_zero_on_four_sources
Tests that the gradient is zero on four sources after joint optimization.
Makes sure that with fewer iterations the gradient is not zero.
"""
function test_gradient_is_near_zero_on_four_sources(; use_fft=false)
    box = BoundingBox(154.39, 164.41, 39.11, 39.13)
    field_triplets = [RunCamcolField(3900, 6, 269),]

    catalog, target_sources = infer_init(field_triplets, datadir; box=box)
    images = SDSSIO.load_field_images(field_triplets, datadir)

    infer_few(ctni...) = one_node_joint_infer(ctni...;
                                              n_iters=2,
                                              within_batch_shuffling=true,
                                              use_fft=use_fft)
    result_few = one_node_infer(field_triplets, datadir;
                                infer_callback=infer_few,
                                box=box)
    @test !unconstrained_gradient_near_zero(target_sources, catalog, images, [x.vs for x in result_few])

    infer_many(ctni...) = one_node_joint_infer(ctni...;
                                               n_iters=100,
                                               within_batch_shuffling=true,
                                               use_fft=use_fft)
    result_many = one_node_infer(field_triplets, datadir;
                                 infer_callback=infer_many,
                                 box=box)
    @test unconstrained_gradient_near_zero(target_sources, catalog, images, [x.vs for x in result_many])

end

"""
test_different_result_with_different_iter()
Using 3 iters instead of 1 iters should result in a different set of parameters.
"""
function test_different_result_with_different_iter(; use_fft=false)
    # This bounding box has overlapping stars. (neighbor map is not empty)
    box = BoundingBox(154.39, 164.41, 39.11, 39.13)
    field_triplets = [RunCamcolField(3900, 6, 269),]

    infer_few(ctni...) = one_node_joint_infer(ctni...;
                                              n_iters=1,
                                              within_batch_shuffling=false,
                                              use_fft=use_fft)
    result_iter_1 = one_node_infer(field_triplets, datadir;
                                   infer_callback=infer_few, box=box)

    infer_many(ctni...) = one_node_joint_infer(ctni...;
                                               n_iters=5,
                                               within_batch_shuffling=false,
                                               use_fft=use_fft)
    result_iter_5 = one_node_infer(field_triplets, datadir;
                                   infer_callback=infer_many, box=box)

    # Make sure that parameters are not the same
    @test !compare_vp_params(result_iter_1, result_iter_5)
end

"""
test_same_result_with_diff_batch_sizes
Varying batch sizes using the cyclades algorithm should not change final objective value.
"""
function test_same_result_with_diff_batch_sizes(;use_fft=false)
    # This bounding box has overlapping stars. (neighbor map is not empty)
    box = BoundingBox(154.39, 164.41, 39.11, 39.13)
    field_triplets = [RunCamcolField(3900, 6, 269),]

    # With batch size = 7
    infer_few(ctni...) = one_node_joint_infer(ctni...;
                                n_iters=3,
                                batch_size=7,
                                within_batch_shuffling=false,
                                use_fft=use_fft)
    result_bs_7 = one_node_infer(field_triplets, datadir;
                                 infer_callback=infer_few,
                                 box=box)

    # With batch size = 39
    infer_many(ctni...) = one_node_joint_infer(ctni...;
                                n_iters=3,
                                batch_size=39,
                                within_batch_shuffling=false,
                                use_fft=use_fft)
    result_bs_39 = one_node_infer(field_triplets, datadir;
                                  infer_callback=infer_many,
                                  box=box)

    # Make sure that parameters are exactly the same
    @test compare_vp_params(result_bs_7, result_bs_39)
end

"""
test infer multi iter with a single (run, camcol, field).
This is basically just to make sure it runs at all.
"""
function test_one_node_joint_infer()
    # very small patch of sky that turns out to have 4 sources.
    # We checked that this patch is in the given field.
    box = BoundingBox(164.39, 164.41, 39.11, 39.13)
    field_triplets = [RunCamcolField(3900, 6, 269),]
    result = one_node_infer(field_triplets, datadir;
                            infer_callback=one_node_joint_infer,
                            box=box)
end


"""
test infer multi iter obj overlapping.
This makes sure one_node_joint_infer achieves sum objective value less
than single iter on non overlapping sources.
todo(max) - this fails with use_fft=true. Fix.
"""
function test_one_node_joint_infer_obj_overlapping(;use_fft=false)

    # This bounding box has overlapping stars. (neighbor map is not empty)
    box = BoundingBox(154.39, 164.41, 39.11, 39.13)
    field_triplets = [RunCamcolField(3900, 6, 269),]

    # 100 iterations
    tic()
    infer_multi(ctni...) = one_node_joint_infer(ctni...;
                                                n_iters=30,
                                                within_batch_shuffling=true,
                                                use_fft=use_fft)
    result_multi = one_node_infer(field_triplets, datadir;
                                  infer_callback=infer_multi,
                                  box=box)
    multi_iter_time = toq()
    score_multi = compute_obj_value(result_multi, field_triplets, datadir; box=box)

    # 2 iterations
    tic()
    infer_two(ctni...) = one_node_joint_infer(ctni...;
                                              n_iters=2,
                                              within_batch_shuffling=true,
                                              use_fft=use_fft)
    result_two = one_node_infer(field_triplets, datadir;
                                infer_callback=infer_two,
                                box=box)
    two_iter_time = toq()
    score_two = compute_obj_value(result_two, field_triplets, datadir; box=box)

    # One node infer (1 iteration, butm ore newton steps)
    tic()
    infer_source_callback = use_fft ?
        DeterministicVIImagePSF.infer_source_fft :
        DeterministicVI.infer_source
    infer_single(ctni...) = one_node_single_infer(ctni...;
                                                  infer_source_callback=infer_source_callback)
    result_single = one_node_infer(field_triplets, datadir;
                                   infer_callback=infer_single,
                                   box=box)
    single_iter_time = toq()
    score_single = compute_obj_value(result_single, field_triplets, datadir; box=box)

    println("One node joint infer objective value: $(score_multi)")
    println("One node joint infer 2 iter objective value: $(score_two)")
    println("Single iter objective value: $(score_single)")
    println("One node joint infer time: $(multi_iter_time)")
    println("One node joint infer 2 iter time: $(two_iter_time)")
    println("Single iter time: $(single_iter_time)")
    @test score_multi > score_single
    @test score_multi > score_two
end


"""
helper edges_between_sources
Helper function to determine if there are edges between 2 sets of sources
"""
function edges_between_sources(a, b, neighbor_map)
    for v_a in a, v_b in b
        # We assume undirected graph (symmetric map)
        if v_b in neighbor_map[v_a]
            println("Found edge between elements $(v_a) and $(v_b)")
            return true
        end
    end
    return false
end


"""
test cyclades partitioning.
Makes sure cyclades partitioning correctly partitions sources
"""
function test_cyclades_partitioning()
    """
    The conflict pattern is a connected 4x4 grid:
    1 -  2 - 3  - 4
    |    |   |    |
    5 -  6 - 7  - 8
    |    |   |    |
    9 - 10 - 11 - 12
    |    |   |    |
   13 - 14 - 15 - 16

    We offset 5 from every value.
    """
    neighbor_map = Dict(6=>[7,10],
                        7=>[6,8,14],
                        8=>[7,9,12],
                        9=>[8,13],
                        10=>[6,11,14],
                        11=>[7,10,12,15],
                        12=>[8,11,13,16],
                        13=>[9,12,17],
                        14=>[11,15,18],
                        15=>[11,14,16,19],
                        16=>[12,15,17,20],
                        17=>[13,16,21],
                        18=>[14,19],
                        19=>[18,15,20],
                        20=>[16,19,21],
                        21=>[17,20])

    target_sources = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]

    # 3 threads, 16 sources, batch_size = 4
    source_assignment = ParallelRun.partition_cyclades(3,
                                            target_sources,
                                            neighbor_map,
                                            batch_size=4)

    # Gather the entities across all batches.
    # 1. They should sum up to 16 exactly.
    # 2. They should be numbered from 1 - 16.
    all_sources =  Vector{Int64}()
    n_simulated_threads = length(source_assignment)
    n_batches = length(source_assignment[1])
    @test n_simulated_threads == 3
    for simulated_thread_id=1:3
        # Every thread must have the same # of batches.
        @test n_batches == length(source_assignment[simulated_thread_id])
        for batch=1:n_batches
            for source_id in source_assignment[simulated_thread_id][batch]
                push!(all_sources, source_id)
            end
        end
    end
    @test length(all_sources) == 16
    for source_id=1:16
        @test source_id in all_sources
    end

    # Now make sure elements of the same batches between threads do not
    # conflict (have an edge).
    for batch=1:n_batches
        t1_sources = [target_sources[x] for x in source_assignment[1][batch]]
        t2_sources = [target_sources[x] for x in source_assignment[2][batch]]
        t3_sources = [target_sources[x] for x in source_assignment[3][batch]]
        println("Batch $(batch) assignments: $t1_sources $t2_sources $t3_sources")
        @test !edges_between_sources(t1_sources, t2_sources, neighbor_map)
        @test !edges_between_sources(t1_sources, t3_sources, neighbor_map)
        @test !edges_between_sources(t2_sources, t3_sources, neighbor_map)
    end

    println("Cyclades partitioning test succeeded")
end

# Stripe 82 tests are long running
if test_long_running
    test_improve_stripe_82_obj_value()

    # Test gradients near zero (this takes 100 iterations and is a bit slow)
    test_gradient_is_near_zero_on_four_sources(; use_fft=false)

    # Test that we reach a higher objective with more iterations. This is a bit slow.
    test_one_node_joint_infer_obj_overlapping()
end

# Test fft is working
test_fft_on_one_source_matches_single()

# Test with using fft
#test_different_result_with_different_iter(use_fft=true)
#test_same_result_with_diff_batch_sizes(use_fft=true)
#test_one_node_joint_infer_obj_overlapping(use_fft=true)

# Test non fft
test_different_result_with_different_iter()
test_same_result_with_diff_batch_sizes()

# Run this multiple times, since the cyclades algorithm shuffles the elements
# before batching them up.
for i=1:20
    test_cyclades_partitioning()
end

test_one_node_joint_infer()
