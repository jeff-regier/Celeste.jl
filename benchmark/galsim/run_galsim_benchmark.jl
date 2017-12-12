#!/usr/bin/env julia

# This script sometimes (but not always) crashes Julia v0.6.0, first triggering the warning:
#     WARNING: An error occurred during inference. Type inference is now partially disabled.
#     Base.MethodError(f=typeof(Core.Inference.convert)(), args=(Base.AssertionError, "invalid age range update"), world=0x0000000000000ac2)
# The issue resembles
#     https://github.com/JuliaLang/julia/issues/23200
#     https://github.com/JuliaLang/julia/issues/22355

import Celeste.AccuracyBenchmark
import Celeste.GalsimBenchmark

test_case_names = String[]
for arg in ARGS
    if ismatch(r"^--test_case_names", arg)
        # test case name: function name in test_case_definitions.py, or CL_DESCR
        # field from FITS header
        test_case_names =
            split(replace(arg, "--test_case_names=", ""), ",")
    else
        println("Usage: run_galsim_benchmark.jl " *
                "[--test_case_names=test1,test2,...]")
        error("Invalid argument: ", arg)
    end
end

srand(12345)

truth_catalog, single_predictions = GalsimBenchmark.run_benchmarks(
    test_case_names=test_case_names,
    joint_inference=false,
    verbose=true)
unused, joint_predictions = GalsimBenchmark.run_benchmarks(
    test_case_names=test_case_names,
    joint_inference=true,
    verbose=true)

score_df = AccuracyBenchmark.score_predictions(
    truth_catalog,
    [single_predictions, joint_predictions],
)

println("\n\n=========================\n")
println("overall scores:\n")
display(score_df)
println()
