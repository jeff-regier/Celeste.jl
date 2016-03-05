#!/usr/bin/env julia

using Celeste
using CelesteTypes
using DocOpt
using JLD
import SloanDigitalSkySurvey: SDSS
import Base: convert

const DOC =
"""Run Celeste.

Usage:
  celeste.jl infer <dir> <run> <camcol> <field> [--outdir=<outdir> --part=<k/n>]
  celeste.jl score <dir> <run> <camcol> <field> <reffile> [--outdir=<outdir>]
  celeste.jl -h | --help
  celeste.jl --version

Options:
  -h, --help         Show this screen.
  --version          Show the version.
  --outdir=<outdir>  Write output files for each source to this directory.
                     Default is to write to input directory. When scoring,
                     this is where celeste result files are read *from*.
  --part=<k/n>       In inferences, split sources into `n` parts and only
                     process the `k`th part.

The `infer` step creates files in the output directory `<outdir>` with
filename format `celeste-<run>-<camcol>-<field>--part-K-M.jld`. The ability
to process only subsets of sources in a run/camcol/field is simply a
low-effort way to parallelize and to run shorter jobs.

In the `infer` step, `<outdir>` is searched for output files matching
`celeste-<run>-<camcol>-<field>-*.jld` and all sources are
concatenated.  A \"truth\" catalog from `<reffile>`, a FITS file
manually created by running a CasJobs query on the Stripe82 database,
based on RA/Dec (see Celeste README for the query). It is expected
that every Celeste source should have a match in this \"truth\"
catalog, otherwise an error is thrown. So, we need to use (run,
camcol, field) combinations fall entirely within the RA/Dec patch used
in that query.
"""


"""
Parse a string like \"1/3\" and return (1, 3).
"""
function parse_part(s)
    words = split(s, '/', keep=false)
    if length(words) == 2
        return (parse(Int, words[1]), parse(Int, words[2]))
    else
        error("part must be two integers separated by `/`.")
    end
end


function main()
    args = docopt(DOC, version=v"0.0.0")

    if args["infer"] || args["score"]
        dir = args["<dir>"]
        run = @sprintf "%06d" parse(Int, args["<run>"])
        camcol = string(parse(Int, args["<camcol>"]))
        field = @sprintf "%04d" parse(Int, args["<field>"])
        outdir = (args["--outdir"] === nothing)? dir: args["--outdir"]

        if args["infer"]
            outdir = (args["--outdir"] === nothing)? dir: args["--outdir"]
            part = (args["--part"] === nothing)? "1/1": args["--part"]
            partnum, parts = parse_part(part)
            Celeste.infer(dir, run, camcol, field, outdir, partnum, parts)

        elseif args["score"]
            scores = Celeste.score_field(dir, run, camcol, field, outdir,
                                         args["<reffile>"])
            println(scores)

        end
    end
end


main()
