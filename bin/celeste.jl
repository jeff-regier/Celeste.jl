#!/usr/bin/env julia

using DocOpt
import Celeste

const DOC =
"""Run Celeste.

Usage:
  celeste.jl [options] stage-box <ramin> <ramax> <decmin> <decmax>
  celeste.jl [options] infer-box <ramin> <ramax> <decmin> <decmax> <outdir>
  celeste.jl [options] infer-field <run> <camcol> <field> <outdir>
  celeste.jl [options] infer-object <run> <camcol> <field> <objid> <outdir>
  celeste.jl [options] score-field <run> <camcol> <field> <resultdir> <truthfile>
  celeste.jl [options] score-object <run> <camcol> <field> <objid> <resultdir> <truthfile>
  celeste.jl -h | --help
  celeste.jl --version

Options:
  -h, --help         Show this screen.
  --version          Show the version.
  --logging=<LEVEL>  Level for the Logging package (OFF, DEBUG, INFO, WARNING, ERROR, or CRITICAL). [default: WARNING]
  --stage            In `infer-box`, automatically stage files. Default: false.

The `stage-box` subcommand copies and/or uncompresses all files covering the
given RA/Dec range from /project to user's SCRATCH directory.

The `infer-box` subcommand runs Celeste on all sources in the given
RA/Dec range, using all available images. It depends on `stage-box` having
already been run on the same patch of sky.

The `infer-field` subcommand runs Celeste on all sources in the given
field, using only that field.

The `score-field` subcommand is not yet implemented for the new API.
"""

function main()
    Celeste.set_logging_level("WARNING")
    args = docopt(DOC, version=v"0.1.0", options_first=true)
    Celeste.set_logging_level(args["--logging"])
    if args["stage-box"] || args["infer-box"]
        ramin = parse(Float64, args["<ramin>"])
        ramax = parse(Float64, args["<ramax>"])
        decmin = parse(Float64, args["<decmin>"])
        decmax = parse(Float64, args["<decmax>"])
        if args["stage-box"]
            Celeste.stage_box_nersc(ramin, ramax, decmin, decmax)
        elseif args["infer-box"]
            outdir = args["<outdir>"]
            Celeste.infer_box_nersc(ramin, ramax, decmin, decmax, outdir;
                                    stage=args["--stage"])
        end
    else
        run = parse(Int, args["<run>"])
        camcol = parse(Int, args["<camcol>"])
        field = parse(Int, args["<field>"])
        if args["infer-field"]
            outdir = args["<outdir>"]
            Celeste.infer_field_nersc(run, camcol, field, outdir)
        elseif args["infer-object"]
            outdir = args["<outdir>"]
            objid = args["<objid>"]
            Celeste.infer_field_nersc(run, camcol, field, outdir; objid=objid)
        elseif args["score-field"]
            resultdir = args["<resultdir>"]
            truthfile = args["<truthfile>"]
            Celeste.score_field_nersc(run, camcol, field, resultdir, truthfile)
        elseif args["score-object"]
            objid = args["<objid>"]
            resultdir = args["<resultdir>"]
            truthfile = args["<truthfile>"]
            Celeste.score_object_nersc(run, camcol, field, objid, resultdir, truthfile)
        end
    end
end


main()
