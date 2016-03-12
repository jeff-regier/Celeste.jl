#!/usr/bin/env julia

using DocOpt
import Celeste

const DOC =
"""Run Celeste.

Usage:
  celeste.jl infer single <run> <camcol> <field> <objid> <dir> [--outdir=<outdir>]
  celeste.jl score <dir> <run> <camcol> <field> <reffile> [--outdir=<outdir>]
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

"""
Query the SDSS database for all fields that overlap the given RA, Dec range.
"""
function query_ra_dec_range(ramin, ramax, decmin, decmax)

    # The ramin, ramax, etc is a bit unintuitive because we're looking
    # for any overlap.
    q = """select distinct run, camcol, field, ramin, ramax, decmin, decmax
    from frame
    where
      rerun = 301 and
      ramax > $(ramin) and ramin < $(ramax) and
      decmax > $(decmin) and decmin < $(decmax)
    order by run"""

    iobuf = Celeste.SDSSIO.query_sql(q)
    seekstart(iobuf)
    data, colnames = readcsv(iobuf; header=true)

    out = Dict{ASCIIString, Vector}()
    out["run"] = Vector{Int}(data[:, 1])
    out["camcol"] = Vector{Int}(data[:, 2])
    out["field"] = Vector{Int}(data[:, 3])
    out["ramin"] = data[:, 4]
    out["ramax"] = data[:, 5]
    out["decmin"] = data[:, 6]
    out["decmax"] = data[:, 7]

    return out
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


#const S82_INDEX_DIR = "/project/projectdirs/dasrepo/celeste-sc16/S82-index"

    



#main()
