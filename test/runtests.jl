#!/usr/bin/env julia

using Celeste
using CelesteTypes
using Base.Test
using Distributions
using SampleData
using Transform
import Synthetic

include(joinpath(Pkg.dir("Celeste"), "test", "test_sdss.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "test_misc.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "test_wcs.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "test_kl.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "test_constraints.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "test_elbo_values.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "test_derivs.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "test_optimization.jl"))
