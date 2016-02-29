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

const TILE_WIDTH = 20
const MAX_ITERS = 20

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

function infer(dir, run, camcol, field, outdir, partnum, parts)

    # get images
    images = SkyImages.load_sdss_blob(dir, run, camcol, field;
                                      mask_planes=[])

    # load catalog and convert to Array of `CatalogEntry`s.
    cat_df = SDSS.load_catalog_df(dir, run, camcol, field)
    cat_entries = SkyImages.convert_catalog_to_celeste(cat_df, images)

    # limit to just the part of the catalog specified.
    partsize = length(cat_entries) / parts
    minidx = round(Int, partsize*(partnum-1)) + 1
    maxidx = round(Int, partsize*partnum)
    cat_entries = cat_entries[minidx:maxidx]

    # initialize tiled images and model parameters
    tiled_blob, mp = ModelInit.initialize_celeste(images, cat_entries,
                                                  tile_width=TILE_WIDTH,
                                                  fit_psf=true)

    # Initialize output dictionary
    nsources = length(minidx:maxidx)
    out = Dict("obj" => minidx:maxidx,  # index within field
               "objid" => Array(ASCIIString, nsources),
               "vp" => Array(Vector{Float64}, nsources),
               "fit_time"=> Array(Float64, nsources))

    # Loop over sources in model
    for i in 1:mp.S
        println("Processing source $i.")

        mp_s = deepcopy(mp);
        mp_s.active_sources = [i]

        # TODO: This is slow but would run much faster if you had run
        # limit_to_object_data() first.
        trimmed_tiled_blob = ModelInit.trim_source_tiles(i, mp_s, tiled_blob;
                                                         noise_fraction=0.1);

        fit_time = time()
        iter_count, max_f, max_x, result =
            OptimizeElbo.maximize_f(ElboDeriv.elbo, trimmed_tiled_blob, mp_s;
                                    verbose=true, max_iters=MAX_ITERS)
        fit_time = time() - fit_time

        out["objid"][i] = mp_s.objids[i]
        out["vp"][i] = mp_s.vp[i]
        out["fit_time"][i] = fit_time
    end

    outfile = "$outdir/celeste-$run-$camcol-$field--part-$partnum-$parts.jld"
    JLD.save(outfile, out)
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
            infer(dir, run, camcol, field, outdir, partnum, parts)

        elseif args["score"]
            scores = Celeste.score_field(dir, run, camcol, field, outdir,
                                         args["<reffile>"])
            println(scores)

        end
    end
end


main()
