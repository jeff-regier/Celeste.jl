# Run celeste interactively.

using Celeste

using DataFrames
import FITSIO
import JLD
import SloanDigitalSkySurvey: SDSS

using Celeste.Types
import Celeste.SkyImages
import Celeste.ModelInit
import Celeste.OptimizeElbo

dat_dir = joinpath(Pkg.dir("Celeste"), "test/data")

run = "004263"
camcol = "5"
field = "0117"

# run = "003900"
# camcol = "6"
# field = "0269"

#Celeste.infer(dat_dir, run, camcol, field, "/tmp/", 4, 100)

images = SkyImages.load_sdss_blob(dat_dir, run, camcol, field);

# load catalog and convert to Array of `CatalogEntry`s.
cat_df = SDSS.load_catalog_df(dir, run, camcol, field);
cat_entries = SkyImages.convert_catalog_to_celeste(cat_df, images);

objid = "1237663784734359622"
s = findfirst(objid, cat_df[:objid])

# initialize tiled images and model parameters
tiled_blob, mp = ModelInit.initialize_celeste(images, cat_entries,
                                              tile_width=20,
                                              fit_psf=true);

mp_s = deepcopy(mp);

# Loop over sources in model
println("Processing source $i, objid $(mp.objids[i])")

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
