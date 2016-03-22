#!/usr/bin/env julia

using Celeste: Types, SampleData, Transform
import Celeste: Synthetic, SDSS, WCSUtils

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

if length(ARGS) > 0
    testfiles = ["test_$(arg).jl" for arg in ARGS]
else
    testfiles = ["test_derivatives.jl",
                 "test_elbo_values.jl",
                 "test_psf.jl",
                 "test_images.jl",
                 "test_kl.jl",
                 "test_misc.jl",
                 "test_optimization.jl",
                 "test_sdssio.jl",
                 "test_sensitive_float.jl",
                 "test_transforms.jl",
                 "test_wcs.jl",
#                 "test_sdss.jl",
                 "test_score.jl",
                 "test_infer.jl"]
end

for testfile in testfiles
    try
        include(testfile)
        println("\t\033[1m\033[32mPASSED\033[0m: $(testfile)")
    catch e
        anyerrors = true
        println("\t\033[1m\033[31mFAILED\033[0m: $(testfile)")
        rethrow()  # Fail fast.
    end
end
