#!/usr/bin/env julia

import Celeste.GalsimBenchmark
import Celeste.DeterministicVI: infer_source
import Celeste.DeterministicVIImagePSF: infer_source_fft

test_case_names = String[]
infer_source_callback = infer_source
for arg in ARGS
    if arg == "--fft"
        infer_source_callback = infer_source_fft
    elseif ismatch(r"^--test_case_names", arg)
        # test case name: function name in test_case_definitions.py, or CL_DESCR
        # field from FITS header
        test_case_names =
            split(replace(arg, "--test_case_names=", ""), ",")
    else
        println("Usage: run_galsim_benchmark.jl " *
                "[--fft] [--test_case_names=test1,test2,...]")
        error("Invalid argument: ", arg)
    end
end

GalsimBenchmark.main(
    test_case_names=test_case_names,
    infer_source_callback=infer_source_callback)
