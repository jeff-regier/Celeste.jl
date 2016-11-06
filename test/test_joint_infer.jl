import JLD

"""
test infer multi iter with a single (run, camcol, field).
This is basically just to make sure it runs at all.
"""
function test_one_node_joint_infer()
    # very small patch of sky that turns out to have 4 sources.
    # We checked that this patch is in the given field.
    box = ParallelRun.BoundingBox(164.39, 164.41, 39.11, 39.13)
    field_triplets = [RunCamcolField(3900, 6, 269),]
    result = ParallelRun.one_node_infer(field_triplets, datadir; box=box, joint_infer=true)
end

"""
test infer multi iter obj overlapping.
This makes sure one_node_joint_infer achieves sum objective value lest than single iter on non overlapping sources. 
"""
function test_one_node_joint_infer_obj_overlapping()

    # This bounding box has overlapping stars. (neighbor map is not empty)
    box = ParallelRun.BoundingBox(164.39, 164.41, 39.11, 39.13)
    field_triplets = [RunCamcolField(3900, 6, 269),]
    tic()
    result_multi, obj_values_multi = ParallelRun.one_node_infer(field_triplets, datadir; box=box, joint_infer_n_iters=100, joint_infer=true)
    multi_iter_time = toq()
    tic()
    result_multi, obj_values_two = ParallelRun.one_node_infer(field_triplets, datadir; box=box, joint_infer_n_iters=2, joint_infer=true)
    multi_iter_one_iter_time = toq()
    tic()
    result_single, obj_values_single = ParallelRun.one_node_infer(field_triplets, datadir; box=box)
    single_iter_time = toq()

    sum_multi = sum(obj_values_multi)
    sum_single = sum(obj_values_single)
    sum_two = sum(obj_values_two)
    println("One node joint infer objective value: $(sum_multi)")
    println("One node joint infer 1 iter objective value: $(sum_two)")
    println("Single iter objective value: $(sum_single)")
    println("One node joint infer time: $(multi_iter_time)")
    println("One node joint infer 1 iter time: $(multi_iter_one_iter_time)")
    println("Single iter time: $(single_iter_time)")
    @test sum_multi > sum_single
    @test sum_multi > sum_two
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
    source_assignment = ParallelRun.partition_cyclades(3, target_sources, neighbor_map, batch_size=4)

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

    # Now make sure elements of the same batches between threads do not conflict (have an edge).
    for batch=1:n_batches
        t1_sources = [target_sources[x] for x in source_assignment[1][batch]]
        t2_sources = [target_sources[x] for x in source_assignment[2][batch]]
        t3_sources = [target_sources[x] for x in source_assignment[3][batch]]
        println("Batch $(batch) assignments")
        println(t1_sources)
        println(t2_sources)
        println(t3_sources)
        @test !edges_between_sources(t1_sources, t2_sources, neighbor_map)
        @test !edges_between_sources(t1_sources, t3_sources, neighbor_map)
        @test !edges_between_sources(t2_sources, t3_sources, neighbor_map)
    end

    println("Cyclades partitioning test succeeded")
end

# Run this multiple times, since the cyclades algorithm shuffles the elements
# before batching them up.
for i=1:20
    test_cyclades_partitioning()
end

test_one_node_joint_infer()
test_one_node_joint_infer_obj_overlapping()
