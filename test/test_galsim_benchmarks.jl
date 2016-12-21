using Base.Test

using DataFrames

import Celeste: GalsimBenchmark

GALSIM_CASES_EXERCISED = [
    #"simple_star", "probability of galaxy" is wrong (#482)
    "star_with_noise",
    "angle_and_axis_ratio_1",
    "galaxy_with_all",
    #"galaxy_with_noise", half-light radius and brightness are wrong (#482)
]

@testset "GalSim benchmark tests" begin
    results = GalsimBenchmark.main(
        test_case_names=GALSIM_CASES_EXERCISED,
        print_fn=x -> 0,
    )
    for row_index in 1:size(results, 1)
        row = results[row_index, :]
        if isna(row[1, :ground_truth])
            continue
        end
        if row[1, :field] == "Probability of galaxy"
            maximum_error = 0.1
        elseif row[1, :field] == "Angle (degrees)"
            maximum_error = 5
        else
            maximum_error = 0.1 * max(row[1, :ground_truth])
        end
        for result_column in [:single_inferred, :joint_inferred]
            @test_approx_eq_eps(row[1, :ground_truth], row[1, result_column], maximum_error)
        end
    end
end
