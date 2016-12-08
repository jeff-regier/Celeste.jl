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
        if isna(row[1, :expected])
            continue
        end
        for result_column in [:single_infer_actual, :joint_infer_actual]
            @test_approx_eq_eps(
                results[1, :expected],
                results[1, result_column],
                0.05 * results[1, :expected], # maximum permissible error
            )
        end
    end
end
