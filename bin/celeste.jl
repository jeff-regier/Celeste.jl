#!/usr/bin/env julia

using DocOpt
import Celeste

const DOC =
"""Run Celeste.

Usage:
  celeste.jl [options] infer-box <ramin> <ramax> <decmin> <decmax> <outdir>
  celeste.jl [options] infer-field <run> <camcol> <field> <outdir>
  celeste.jl [options] score-field <run> <camcol> <field> <resultdir> <truthfile>
  celeste.jl -h | --help
  celeste.jl --version

Options:
  -h, --help         Show this screen.
  --version          Show the version.
  --logging=<LEVEL>   Level for the Logging package (OFF, DEBUG, INFO, WARNING, ERROR, or CRITICAL). [default: INFO]

The `infer-box` subcommand runs Celeste on all sources in the given
RA/Dec range, using all available images.

The `infer-field` subcommand runs Celeste on all sources in the given
field, using only that field.

The `score-field` subcommand is not yet implemented for the new API.
"""

function main()
    args = docopt(DOC, version=v"0.1.0", options_first=true)
    Celeste.set_logging_level(args["--logging"])
    if args["infer-box"]
        ramin = parse(Float64, args["<ramin>"])
        ramax = parse(Float64, args["<ramax>"])
        decmin = parse(Float64, args["<decmin>"])
        decmax = parse(Float64, args["<decmax>"])
        outdir = args["<outdir>"]
        Celeste.infer_box_nersc(ramin, ramax, decmin, decmax, outdir)
    elseif args["infer-field"]
        run = parse(Int, args["<run>"])
        camcol = parse(Int, args["<camcol>"])
        field = parse(Int, args["<field>"])
        outdir = args["<outdir>"]
        Celeste.infer_field_nersc(run, camcol, field, outdir)
    elseif args["score-field"]
        Celeste.score_field_nersc(args["<resultdir>"], args["<truthfile>"])
    end
end


main()
