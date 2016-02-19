#!/usr/bin/env julia

using Celeste
using CelesteTypes
using DocOpt
import JLD
import SloanDigitalSkySurvey: SDSS

const DOC =
"""Run Celeste.

Usage:
  celeste.jl fit <dir> <run> <camcol> <field> [--outdir=<outdir> --part=<k/n>]
  celeste.jl score <dir> <run> <camcol> <field>
  celeste.jl -h | --help
  celeste.jl --version

Options:
  -h, --help         Show this screen.
  --version          Show the version.
  --outdir=<outdir>  Write output files for each source to this directory.
                     Default is to write to input directory.
  --part=<k/n>       Split sources into `n` parts and only process the
                     `k`th part.
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


function fit(dir, run, camcol, field, outdir, partnum, parts)

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

    # Loop over sources in model
    for i in 1:mp.S
        println("Processing source $i.")

        mp_s = deepcopy(mp);
        mp_s.active_sources = [i]

        # Some sanity checks and improved initialization.
        objid = mp.objids[i]
        cat_row = cat_df[:objid] .== objid;
        is_star = cat_df[cat_row, :is_star][1]
        mp_s.vp[i][ids.a[1]] = is_star ? 0.8: 0.2
        mp_s.vp[i][ids.a[2]] = 1.0 - mp_s.vp[i][ids.a][1]

        # Skip dim sources
        if cat_df[cat_row, :psfflux_r][1] < 10
            continue
        end

        transform = Transform.get_mp_transform(mp_s);

        # TODO: This is slow but would run much faster if you had run
        # limit_to_object_data() first.  Currently that requires the original
        # blob which cannot be saved to an HDF5 file.
        trimmed_tiled_blob = ModelInit.trim_source_tiles(i, mp_s, tiled_blob;
                                                         noise_fraction=0.1);

        fit_time = time()
        iter_count, max_f, max_x, result =
            OptimizeElbo.maximize_f(ElboDeriv.elbo, trimmed_tiled_blob, mp_s;
                                    verbose=true, max_iters=MAX_ITERS)
        fit_time = time() - fit_time

        outfile = "$outdir/celeste-$run-$camcol-$field-$i.jld"
        JLD.save(outfile, Dict("i" => i,
                               "vp[i]" => mp_s.vp[i],
                               "result" => result,
                               "fit_time" => fit_time))
    end
end


"""
Score all the celeste results for sources in the given (run, camcol, field).
This is done by finding all files with names matching
`dir/celeste-RUN-CAMCOL-FIELD-[0-9]*.jld`
"""
function score(dir, run, camcol, field)
    println("score not yet implemented")
end


function main()
    args = docopt(DOC, version=v"0.0.0")

    if args["fit"] || args["score"]
        dir = args["<dir>"]
        run = @sprintf "%06d" parse(Int, args["<run>"])
        camcol = string(parse(Int, args["<camcol>"]))
        field = @sprintf "%04d" parse(Int, args["<field>"])

        if args["fit"]
            outdir = (args["--outdir"] === nothing)? dir: args["--outdir"]
            part = (args["--part"] === nothing)? "1/1": args["--part"]
            partnum, parts = parse_part(part)
            fit(dir, run, camcol, field, outdir, partnum, parts)
        elseif args["score"]
            score(dir, run, camcol, field)
        end
    end
end


main()
