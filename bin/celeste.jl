#!/usr/bin/env julia

using DocOpt
import Celeste

const DOC =
"""Run Celeste.

Usage:
  celeste.jl infer-nersc <ramin> <ramax> <decmin> <decmax> <outdir> [--logging=<LEVEL>]
  celeste.jl score-nersc <ramin> <ramax> <decmin> <decmax> <resultdir> <reffile> [--logging=<LEVEL>]
  celeste.jl infer <run> <camcol> <field> <objid> <datadir> [--logging=<LEVEL>]
  celeste.jl -h | --help
  celeste.jl --version

Options:
  -h, --help         Show this screen.
  --version          Show the version.
  --logging=<leve>   Level for the Logging package (OFF, DEBUG, INFO, WARNING, ERROR, or CRITICAL).  [ default: INFO ]

The `infer-nersc` subcommand runs Celeste on all sources in the given
RA/Dec range.

The `score-nersc` subcommand is not yet implemented for the new API.
"""


function main()
    args = docopt(DOC, version=v"0.0.0")
    if args["infer-nersc"]
        Celeste.set_logging_level(args["<logging>"])
        ramin = parse(Float64, args["<ramin>"])
        ramax = parse(Float64, args["<ramax>"])
        decmin = parse(Float64, args["<decmin>"])
        decmax = parse(Float64, args["<decmax>"])
        outdir = args["<outdir>"]
        Celeste.infer_nersc(ramin, ramax, decmin, decmax, outdir)
    elseif args["score-nersc"]
        Celeste.set_logging_level(args["<logging>"])
        Celeste.score_nersc(ramargs["<resultdir>"], args["<reffile>"])
    elseif args["infer"]
        Celeste.set_logging_level(args["<logging>"])
        run = parse(Int, args["<run>"])
        camcol = parse(Int, args["<camcol>"])
        field = parse(Int, args["<field>"])
        dir = args["<datadir>"]
        objid = args["<objid>"]
        Celeste.infer(run, camcol, field, objid, dir)
    end

end


main()
