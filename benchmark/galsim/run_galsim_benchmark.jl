#!/usr/bin/env julia

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
)
unused, joint_predictions = GalsimBenchmark.run_benchmarks(
    test_case_names=test_case_names,
    joint_inference=true,
)

score_df = AccuracyBenchmark.score_predictions(
    truth_catalog,
    [single_predictions, joint_predictions],
)
println(repr(score_df))
