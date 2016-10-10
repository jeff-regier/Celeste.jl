#!/usr/bin/env julia

using Celeste: Model, DeterministicVI

import Celeste: Infer, DeterministicVI, ParallelRun
import Celeste: PSF, SDSSIO, SensitiveFloats, Transform
import SensitiveFloats.clear!
import SDSSIO: RunCamcolField

include("Synthetic.jl")
include("SampleData.jl")
include("DerivativeTestUtils.jl")

import Synthetic
using SampleData

using Base.Test
using Distributions

anyerrors = false

# Ensure that test images are available.
const datadir = joinpath(Pkg.dir("Celeste"), "test", "data")
wd = pwd()
cd(datadir)
run(`make`)
run(`make RUN=4263 CAMCOL=5 FIELD=119`)
cd(wd)

# Check whether to run time-consuming derivatives tests.
long_running_flag = "--long-running"
test_long_running = long_running_flag in ARGS
test_files = setdiff(ARGS, [ long_running_flag ])

if length(test_files) > 0
    testfiles = ["test_$(arg).jl" for arg in test_files]
else
    testfiles = ["test_log_prob.jl",
                 "test_elbo.jl",
                 "test_score.jl",
                 "test_derivatives.jl",
                 "test_psf.jl",
                 "test_images.jl",
                 "test_misc.jl",
                 "test_optimization.jl",
                 "test_sdssio.jl",
                 "test_transforms.jl",
                 "test_wcs.jl",
                 "test_infer.jl"]
end


if test_long_running
    warn("Testing ELBO derivatives, which may be slow.")
    push!(testfiles, "test_slow_derivatives.jl")
else
    warn("Skipping long running tests.  ",
         "To test everything, run tests with the flag ", long_running_flag)
end


for testfile in testfiles
    try
        println("Running ", testfile)
        include(testfile)
        println("\t\033[1m\033[32mPASSED\033[0m: $(testfile)")
    catch e
        anyerrors = true
        println("\t\033[1m\033[31mFAILED\033[0m: $(testfile)")
        rethrow()  # Fail fast.
    end
end
