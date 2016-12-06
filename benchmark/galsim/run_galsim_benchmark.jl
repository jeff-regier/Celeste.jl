#!/usr/bin/env julia

include(joinpath(Pkg.dir("Celeste"), "benchmark/galsim/GalsimBenchmark.jl"))
import GalsimBenchmark

if length(ARGS) == 1
    # test case name: function name in test_case_definitions.py, or CL_DESCR field from FITS header
    GalsimBenchmark.main(test_case_name=Nullable(ARGS[1]))
else
    GalsimBenchmark.main()
end
