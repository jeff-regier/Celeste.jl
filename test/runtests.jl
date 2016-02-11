#!/usr/bin/env julia

using Celeste
using CelesteTypes
using Base.Test
using Distributions
using SampleData
using Transform
import Synthetic

anyerrors = false

if length(ARGS) > 0
    testfiles = ["test_$(arg).jl" for arg in ARGS]
else
    testfiles = ["test_derivatives.jl",
                 "test_elbo_values.jl",
                 "test_images.jl",
                 "test_kl.jl",
                 "test_misc.jl",
                 "test_optimization.jl",
                 "test_sensitive_float.jl",
                 "test_transforms.jl"]
end

for testfile in testfiles
    try
        include(testfile)
        println("\t\033[1m\033[32mPASSED\033[0m: $(testfile)")
    catch e
        anyerrors = true
        println("\t\033[1m\033[31mFAILED\033[0m: $(testfile)")
        throw("tests failed")  # Fail fast.
    end
end
