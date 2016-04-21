#!/usr/bin/env julia

using Celeste: Types, Transform
import Celeste: WCSUtils, ModelInit

include("Synthetic.jl")
include("SampleData.jl")

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


"""
Turn a blob and vector of catalog entries into a tiled_blob and model
parameters that can be used with Celeste.
"""
function initialize_celeste(
        blob::Blob, cat::Vector{CatalogEntry};
        tile_width::Int=20, fit_psf::Bool=true,
        patch_radius::Float64=NaN)
    tiled_blob = Types.break_blob_into_tiles(blob, tile_width)
    mp = ModelInit.initialize_model_params(tiled_blob, blob, cat,
                               fit_psf=fit_psf, patch_radius=patch_radius)
    tiled_blob, mp
end


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
