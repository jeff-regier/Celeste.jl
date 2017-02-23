using Base.Test

using DataFrames

import Celeste: GalsimBenchmark

GALSIM_CASES_EXERCISED = [
    #"simple_star",
    "star_with_noise",
    "angle_and_axis_ratio_1",
    "galaxy_with_all",
    #"galaxy_with_noise",
]

function assert_estimates_are_close(benchmark_results)
    for row in eachrow(benchmark_results)
        if isna(row[:truth])
            continue
        end
        if row[:variable] == :is_star
            maximum_error = 0.1
        elseif row[:variable] == :de_vaucouleurs_mixture_weight
            maximum_error = 0.2
        elseif row[:variable] == :angle_deg
            maximum_error = 5
        #elseif !isna(row[1, :error_sds])
        #    maximum_error = 2.5 * row[1, :error_sds]
        else
            maximum_error = 0.2 * abs(row[:truth])
        end
        if !isapprox(row[:truth], row[:estimate], atol=maximum_error)
            @show row
            @show maximum_error
            @test false
        else
            @test true # just so test framework will count test cases
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
