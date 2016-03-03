# Run celeste interactively.

using Celeste
using CelesteTypes
using DocOpt
using JLD
import SloanDigitalSkySurvey: SDSS

dat_dir = joinpath(Pkg.dir("Celeste"), "test/data")

run = "003900"
camcol = "6"
field = "0269"

Celeste.infer(dat_dir, run, camcol, field, "/tmp/", 1, 1000)
