#!/usr/bin/env julia

using Celeste
using CelesteTypes
using DocOpt
import JLD
import SloanDigitalSkySurvey: SDSS

const doc =
"""Run Celeste.

Usage:
  celeste.jl <dir> <run> <camcol> <field> [--outdir=<outdir>]
  celeste.jl -h | --help
  celeste.jl --version
"""
# E.g., celeste.jl ../dat/sample_field 3900 6 269

const TILE_WIDTH = 20
const MAX_ITERS = 20

function main()
    args = docopt(doc, version=v"0.0.0")

    dir = args["<dir>"]
    run = @sprintf "%06d" parse(Int, args["<run>"])
    camcol = string(parse(Int, args["<camcol>"]))
    field = @sprintf "%04d" parse(Int, args["<field>"])

    outdir = (args["--outdir"] === nothing)? dir: args["--outdir"]

    # get images
    images = SkyImages.load_sdss_blob(dir, run, camcol, field;
                                      mask_planes=[])

    # load catalog and convert to Array of `CatalogEntry`s.
    cat_df = SDSS.load_catalog_df(dir, run, camcol, field)
    cat_entries = SkyImages.convert_catalog_to_celeste(cat_df, images)

    # initialize tiled images and model parameters
    tiled_blob, mp = ModelInit.initialize_celeste(images, cat_entries,
                                                  tile_width=TILE_WIDTH,
                                                  fit_psf=false)

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

end # main function

main()
