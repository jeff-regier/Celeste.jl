import JLD
import ..Infer
import ..ParallelRun: infer_init, one_node_infer, BoundingBox,
                      one_node_joint_infer, one_node_single_infer


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
                           box=BoundingBox(-1000.,1000.,-1000.,1000.))
    catalog, target_sources = infer_init(rcfs, stagedir; box=box)
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

"""
test_different_result_with_different_iter()
Using 3 iters instead of 1 iters should result in a different set of parameters.
"""
function test_different_result_with_different_iter()
    # This bounding box has overlapping stars. (neighbor map is not empty)
    box = BoundingBox(4.39, 164.41, 9.11, 39.13)
    field_triplets = [RunCamcolField(3900, 6, 269),]

    infer_few(ctni...) = one_node_joint_infer(ctni...;
                                              n_iters=1,
                                              within_batch_shuffling=false)
    result_iter_1 = one_node_infer(field_triplets, datadir;
                                   infer_callback=infer_few, box=box)

    infer_many(ctni...) = one_node_joint_infer(ctni...;
                                             n_iters=5,
                                             within_batch_shuffling=false)
    result_iter_5 = one_node_infer(field_triplets, datadir;
                                   infer_callback=infer_many, box=box)

    # Make sure that parameters are not the same
    @test !compare_vp_params(result_iter_1, result_iter_5)
end

"""
test_same_result_with_diff_batch_sizes
Varying batch sizes using the cyclades algorithm should not change final objective value.
"""
function test_same_result_with_diff_batch_sizes()
    # This bounding box has overlapping stars. (neighbor map is not empty)
    box = BoundingBox(4.39, 164.41, 9.11, 39.13)
    field_triplets = [RunCamcolField(3900, 6, 269),]

    # With batch size = 7
    infer_few(ctni...) = one_node_joint_infer(ctni...;
                                n_iters=3,
                                batch_size=7,
                                within_batch_shuffling=false)
    result_bs_7 = one_node_infer(field_triplets, datadir;
                                 infer_callback=infer_few,
                                 box=box)

    # With batch size = 39
    infer_many(ctni...) = one_node_joint_infer(ctni...;
                                n_iters=3,
                                batch_size=39,
                                within_batch_shuffling=false)
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
"""
function test_one_node_joint_infer_obj_overlapping()

    # This bounding box has overlapping stars. (neighbor map is not empty)
    box = BoundingBox(164.39, 164.41, 39.11, 39.13)
    field_triplets = [RunCamcolField(3900, 6, 269),]

    # 100 iterations
    tic()
    infer_multi(ctni...) = one_node_joint_infer(ctni...;
                                     n_iters=100,
                                     within_batch_shuffling=false)
    result_multi = one_node_infer(field_triplets, datadir;
                                  infer_callback=infer_multi,
                                  box=box)
    multi_iter_time = toq()
    score_multi = compute_obj_value(result_multi, field_triplets, datadir; box=box)

    # 2 iterations
    tic()
    infer_two(ctni...) = one_node_joint_infer(ctni...;
                                     n_iters=2,
                                     within_batch_shuffling=false)
    result_two = one_node_infer(field_triplets, datadir;
                                  infer_callback=infer_two,
                                  box=box)
    two_iter_time = toq()
    score_two = compute_obj_value(result_two, field_triplets, datadir; box=box)

    # One node infer (1 iteration, butm ore newton steps)
    tic()
    result_single = one_node_infer(field_triplets, datadir; box=box)
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

test_different_result_with_different_iter()
test_same_result_with_diff_batch_sizes()
test_one_node_joint_infer_obj_overlapping()

# Run this multiple times, since the cyclades algorithm shuffles the elements
# before batching them up.
for i=1:20
    test_cyclades_partitioning()
end

test_one_node_joint_infer()
