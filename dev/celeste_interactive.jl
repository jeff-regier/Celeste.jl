# Run celeste interactively.

using Celeste

using DataFrames
using PyPlot

import FITSIO
import JLD
import SloanDigitalSkySurvey: SDSS
import SloanDigitalSkySurvey: WCSUtils


using Celeste.Types
import Celeste.SkyImages
import Celeste.ModelInit
import Celeste.OptimizeElbo
import Celeste.ElboDeriv

dir = joinpath(Pkg.dir("Celeste"), "test/data")

run_string = "003900"
camcol_string = "6"
field_string = "0269"

run = 3900
camcol = 6
field = 269

images = SkyImages.read_sdss_field(run, camcol, field, dir);

# load catalog and convert to Array of `CatalogEntry`s.
cat_df = SDSS.load_catalog_df(dir, run_string, camcol_string, field_string);
cat_filename = "photoObj-$run_string-$camcol_string-$field_string.fits"
cat_entries = SkyImages.read_photoobj_celeste(joinpath(dir, cat_filename));

# initialize tiled images and model parameters.  Don't fit the psf for now --
# we just need the tile_sources from mp.
tiled_blob, mp = ModelInit.initialize_celeste(images, cat_entries,
                                              tile_width=20,
                                              fit_psf=false);

## Look at fluxes
MAX_FLUX = 2
flux_cols = [ symbol("psfflux_$b") for b in band_letters ]
min_fluxes = Float64[ minimum([ cat_df[row, flux_col] for flux_col in flux_cols ])
                      for row in 1:size(cat_df, 1) ];
max_fluxes = Float64[ maximum([ cat_df[row, flux_col] for flux_col in flux_cols ])
                      for row in 1:size(cat_df, 1) ];
#PyPlot.plt[:hist](max_fluxes, 200)
good_rows = max_fluxes .> 3;
sum(good_rows) / length(good_rows)

bad_objids = cat_df[:objid][!good_rows];


# Choose an object:
objid = "1237662226208063491"
s = findfirst(mp.objids, objid)
relevant_sources = ModelInit.get_relevant_sources(mp, s);
ModelInit.fit_object_psfs!(mp, relevant_sources, images);
mp.active_sources = [ s ];

#for objid in bad_objids
trimmed_tiled_blob =
  ModelInit.trim_source_tiles(s, mp, tiled_blob, noise_fraction=0.1);

band = 3
stitched_image, h_range, w_range =
  Celeste.SkyImages.stitch_object_tiles(s, band, mp, trimmed_tiled_blob, predicted=true);

pix_loc = WCSUtils.world_to_pix(
  mp.patches[s, band], mp.vp[s][ids.u])
matshow(stitched_image, vmax=1200);
PyPlot.plot(pix_loc[2] - w_range[1] + 1, pix_loc[1] - h_range[1] + 1, "wo", markersize=5)
PyPlot.colorbar()
PyPlot.title(objid)
# PyPlot.savefig("/tmp/celeste_images/celeste_$objid.png")
# PyPlot.close()

fit_time = time()
iter_count, max_f, max_x, result =
    OptimizeElbo.maximize_f(ElboDeriv.elbo, trimmed_tiled_blob, mp;
                            verbose=true, max_iters=50)
fit_time = time() - fit_time
