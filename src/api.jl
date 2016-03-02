# Functions for interacting with Celeste from the command line.

using Celeste
using CelesteTypes
using DocOpt
using JLD
import SloanDigitalSkySurvey: SDSS

const TILE_WIDTH = 20
const MAX_ITERS = 20

"""
Fit the Celeste model to a set of sources and write the output to a JLD file.

Args:
  dir: The directory containing the FITS files.
  run: An ASCIIstring with the six-digit run number, e.g. "003900"
  camcol: An ASCIIstring with the camcol, e.g. "6"
  field: An ASCIIstring with the four-digit field, e.g. "0269"
  outdir: The directory to write the output jld file.
  partnum: Which of the 1:parts catalog entries to fit.
  parts: How many parts to divide the catalog entries into

Returns:
  Writes a jld file to outdir containing the optimization output.
"""
function infer(dir::ASCIIstring, run::ASCIIstring, camcol::ASCIIstring,
               field::ASCIIstring, outdir::ASCIIstring,
               partnum::Int64, parts::Int64)

    # get images
    images = SkyImages.load_sdss_blob(dir, run, camcol, field)

    # load catalog and convert to Array of `CatalogEntry`s.
    cat_df = SDSS.load_catalog_df(dir, run, camcol, field)
    cat_entries = SkyImages.convert_catalog_to_celeste(cat_df, images)

    # limit to just the part of the catalog specified.
    partsize = length(cat_entries) / parts
    minidx = round(Int, partsize * (partnum - 1)) + 1
    maxidx = round(Int, partsize * partnum)
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
