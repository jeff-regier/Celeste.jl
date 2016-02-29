module Celeste

push!(LOAD_PATH, joinpath(Pkg.dir("Celeste"), "src"))

import SkyImages
import ElboDeriv
import OptimizeElbo
import ModelInit
import SampleData
import Transform
import KL

using CelesteTypes

using DataFrames
using JLD
using FITSIO
import SloanDigitalSkySurvey.SDSS

include("score.jl")

end # module
