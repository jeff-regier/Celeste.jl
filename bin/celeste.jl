#!/usr/bin/env julia

using DocOpt
import Celeste
import Celeste.SDSSIO.RunCamcolField

const DOC =
"""Run Celeste.

Usage:
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
  --logging=<level>  Level for the Lumberjack package (debug, info, warn, error). [default: info]

The `infer-box` subcommand runs Celeste on all sources in the given
RA/Dec range, using all available images.

The `infer-field` subcommand runs Celeste on all sources in the given
field, using only that field.

The `score-field` subcommand is not yet implemented for the new API.
"""

function main()
    args = docopt(DOC, version=v"0.1.0", options_first=true)
#   TODO: re-enable selective logging by level
#    set_logging_level(args["--logging"])
    stagedir = ENV["CELESTE_STAGE_DIR"]
    if args["infer-box"]
        ramin = parse(Float64, args["<ramin>"])
        ramax = parse(Float64, args["<ramax>"])
        decmin = parse(Float64, args["<decmin>"])
        decmax = parse(Float64, args["<decmax>"])
        box = Celeste.BoundingBox(ramin, ramax, decmin, decmax)
        outdir = args["<outdir>"]
        Celeste.infer_box(box, stagedir, outdir)
    else
        run = parse(Int, args["<run>"])
        camcol = parse(Int, args["<camcol>"])
        field = parse(Int, args["<field>"])
        rcf = RunCamcolField(run, camcol, field)
        if args["infer-field"]
            outdir = args["<outdir>"]
            Celeste.infer_field(rcf, stagedir, outdir)
        elseif args["infer-object"]
            outdir = args["<outdir>"]
            objid = args["<objid>"]
            Celeste.infer_field(rcf, stagedir, outdir; objid=objid)
        elseif args["score-field"]
            resultdir = args["<resultdir>"]
            truthfile = args["<truthfile>"]
            Celeste.Stripe82Score.score_field_disk(rcf, resultdir, truthfile)
        elseif args["score-object"]
            objid = args["<objid>"]
            resultdir = args["<resultdir>"]
            truthfile = args["<truthfile>"]
            Celeste.Stripe82Score.score_object_disk(rcf, objid, resultdir, truthfile)
        end
    end
end


main()
