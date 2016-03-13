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
  --outdir=<outdir>  Write output files for each source to this directory.
                     Default is to write to input directory. When scoring,
                     this is where celeste result files are read *from*.

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
