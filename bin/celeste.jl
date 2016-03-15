#!/usr/bin/env julia

using DocOpt
import Celeste

const DOC =
"""Run Celeste.

Usage:
  celeste.jl infer-nersc <ramin> <ramax> <decmin> <decmax>
  celeste.jl score
  celeste.jl -h | --help
  celeste.jl --version

Options:
  -h, --help         Show this screen.
  --version          Show the version.

The `infer-nersc` subcommand runs Celeste on all sources in the given
RA/Dec range.

The `score` step is not yet implemented for the new API.
"""


function main()
    args = docopt(DOC, version=v"0.0.0")
    if args["infer-nersc"]
        ramin = parse(Float64, args["<ramin>"])
        ramax = parse(Float64, args["<ramax>"])
        decmin = parse(Float64, args["<decmin>"])
        decmax = parse(Float64, args["<decmax>"])
        Celeste.infer_nersc(ramin, ramax, decmin, decmax)
    elseif args["score"]
        error("score not yet implemented for infer-nersc")
    end

end


main()
