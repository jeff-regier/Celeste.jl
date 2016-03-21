# Run celeste interactively.

using Celeste

using DataFrames

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
tiled_blob, mp =
  ModelInit.initialize_celeste(images, cat_entries, tile_width=20, fit_psf=false);

# Limit to bright objects
MAX_FLUX = 10
flux_cols = [ symbol("psfflux_$b") for b in band_letters ]
max_fluxes = Float64[ maximum([ cat_df[row, flux_col] for flux_col in flux_cols ])
                      for row in 1:size(cat_df, 1) ];
good_rows = max_fluxes .> 3;

# Choose an object:
objid = "1237662226208063491"
s = findfirst(mp.objids, objid)
relevant_sources = ModelInit.get_relevant_sources(mp, s);
ModelInit.fit_object_psfs!(mp, relevant_sources, images);
mp.active_sources = [ s ];
trimmed_tiled_blob =
  ModelInit.trim_source_tiles(s, mp, tiled_blob, noise_fraction=0.1);

# Perform the actual fit
fit_time = time()
iter_count, max_f, max_x, result =
    OptimizeElbo.maximize_f(ElboDeriv.elbo, trimmed_tiled_blob, mp;
                            verbose=true, max_iters=50)
fit_time = time() - fit_time
