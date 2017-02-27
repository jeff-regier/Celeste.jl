#!/usr/bin/env julia


Pkg.checkout("DataArrays")
Pkg.build("DataArrays")
Pkg.checkout("DataFrames", "anj/06")
Pkg.build("DataFrames")


using Celeste: Model, DeterministicVI

import Celeste: Infer, DeterministicVI, ParallelRun, DeterministicVIImagePSF
import Celeste: PSF, SDSSIO, SensitiveFloats, Transform, CelesteEDA
import SensitiveFloats.clear!
import SDSSIO: RunCamcolField

include(joinpath(Pkg.dir("Celeste"), "test", "Synthetic.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "SampleData.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "DerivativeTestUtils.jl"))

import Synthetic
using SampleData

using Base.Test
using Distributions

anyerrors = false

wd = pwd()
# Ensure that test images are available.
const datadir = joinpath(Pkg.dir("Celeste"), "test", "data")
cd(datadir)
run(`make`)
run(`make RUN=4263 CAMCOL=5 FIELD=119`)
# Ensure GalSim test images are available.
const galsim_benchmark_dir = joinpath(Pkg.dir("Celeste"), "benchmark", "galsim")
cd(galsim_benchmark_dir)
run(`make fetch`)
cd(wd)

# Check whether to run time-consuming derivatives tests.
long_running_flag = "--long-running"
test_long_running = long_running_flag in ARGS
test_files = setdiff(ARGS, [ long_running_flag ])

if length(test_files) > 0
    testfiles = ["test_$(arg).jl" for arg in test_files]
else
    testfiles = [
                 "test_derivatives.jl",
                 "test_kl.jl",
                 "test_eda.jl",
                 "test_constraints.jl",
                 "test_elbo.jl",
                 "test_fft.jl",
                 "test_galsim_benchmarks.jl",
                 "test_images.jl",
                 "test_infer.jl",
                 "test_joint_infer.jl",
                 "test_kernels.jl",
                 "test_log_prob.jl",
                 "test_misc.jl",
                 "test_optimization.jl",
                 "test_psf.jl",
                 "test_score.jl",
                 "test_sdssio.jl",
                 "test_transforms.jl",
                 "test_wcs.jl",
                ]
end


if !test_long_running
    warn("Skipping long running tests.  ",
         "To test everything, run tests with the flag ", long_running_flag)
end


for testfile in testfiles
    try
        println("Running ", testfile)
        @time include(testfile)
        println("\t\033[1m\033[32mPASSED\033[0m: $(testfile)")
    catch e
        anyerrors = true
        println("\t\033[1m\033[31mFAILED\033[0m: $(testfile)")
        rethrow()  # Fail fast.
    end
end
