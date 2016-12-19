using Base.Test

using DataFrames

import Celeste: GalsimBenchmark

@testset "GalSim benchmark tests" begin
    results = GalsimBenchmark.main(
        test_case_names=["simple_star", "angle_and_axis_ratio_1"],
        print_fn=x -> 0,
    )
    for row_index in 1:size(results, 1)
        row = results[row_index, :]
        if isna(row[1, :ground_truth])
            continue
        end
        for result_column in [:single_inferred, :joint_inferred]
            @test_approx_eq_eps(
                results[1, :ground_truth],
                results[1, result_column],
                0.05 * results[1, :ground_truth], # maximum permissible error
            )
        end
    end
end
