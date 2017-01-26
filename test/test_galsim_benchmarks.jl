using Base.Test

import Celeste: GalsimBenchmark

GALSIM_CASES_EXERCISED = [
    "simple_star",
    "star_with_noise",
    "angle_and_axis_ratio_1",
    "galaxy_with_all",
    "galaxy_with_noise",
]

function assert_estimates_are_close(benchmark_results)
    for row_index in 1:size(benchmark_results, 1)
        row = benchmark_results[row_index, :]
        if row[1, :field] == "Probability of galaxy"
            maximum_error = 0.1
        elseif row[1, :field] == "Angle (degrees)"
            maximum_error = 5
        else
            maximum_error = 0.1 * max(row[1, :ground_truth])
        end
        @test isapprox(row[1, :ground_truth], row[1, :estimate], atol=maximum_error)
    end
end

@testset "GalSim benchmark tests, single-source inference" begin
    results = GalsimBenchmark.run_benchmarks(
        test_case_names=GALSIM_CASES_EXERCISED,
        print_fn=x -> 0,
        joint_inference=false
    )
    assert_estimates_are_close(results)
end

@testset "GalSim benchmark tests, joint inference" begin
    results = GalsimBenchmark.run_benchmarks(
        test_case_names=GALSIM_CASES_EXERCISED,
        print_fn=x -> 0,
        joint_inference=true
    )
    assert_estimates_are_close(results)
end
