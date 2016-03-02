# Run celeste interactively.

using Celeste
using CelesteTypes
using DocOpt
using JLD
import SloanDigitalSkySurvey: SDSS

include(joinpath(Pkg.dir("Celeste"), "src/api.jl"))

dat_dir = joinpath(Pkg.dir("Celeste"), "test/data")

run = "003900"
camcol = "6"
field = "0269"

infer(dat_dir, run, camcol, field, "/tmp/", 1, 1000)
