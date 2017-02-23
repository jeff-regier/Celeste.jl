using Base.Test

using DataFrames

import Celeste: GalsimBenchmark

GALSIM_CASES_EXERCISED = [
    #"simple_star",
    "star_with_noise",
    "angle_and_axis_ratio_1",
    "galaxy_with_all",
    "galaxy_with_noise",
]

function assert_estimates_are_close(benchmark_results)
    for row_index in 1:size(benchmark_results, 1)
        row = benchmark_results[row_index, :]
        if row[1, :variable] == :is_star
            maximum_error = 0.1
        elseif row[1, :variable] == :de_vaucouleurs_mixture_weight
            maximum_error = 0.2
        elseif row[1, :variable] == :angle_deg
            maximum_error = 5
        #elseif !isna(row[1, :error_sds])
        #    maximum_error = 2.5 * row[1, :error_sds]
        else
            maximum_error = 0.2 * abs(row[1, :truth])
        end
        if !isapprox(row[1, :truth], row[1, :estimate], atol=maximum_error)
            @show row
            @show maximum_error
            @test false
        end
    end
end

@testset "GalSim benchmark tests, single-source inference" begin
    truth, results = GalsimBenchmark.run_benchmarks(
        test_case_names=GALSIM_CASES_EXERCISED,
        print_fn=x -> 0,
        joint_inference=false
    )
    assert_estimates_are_close(GalsimBenchmark.truth_comparison_df(truth, results))
end

@testset "GalSim benchmark tests, joint inference" begin
    truth, results = GalsimBenchmark.run_benchmarks(
        test_case_names=GALSIM_CASES_EXERCISED,
        print_fn=x -> 0,
        joint_inference=true
    )
    assert_estimates_are_close(GalsimBenchmark.truth_comparison_df(truth, results))
end
