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
import Celeste.ElboDeriv

dir = joinpath(Pkg.dir("Celeste"), "test/data")

run = "004263"
camcol = "5"
field = "0117"

# run = "003900"
# camcol = "6"
# field = "0269"

#Celeste.infer(dir, run, camcol, field, "/tmp/", 4, 100)

images = SkyImages.load_sdss_blob(dir, run, camcol, field);

# load catalog and convert to Array of `CatalogEntry`s.
cat_df = SDSS.load_catalog_df(dir, run, camcol, field);
cat_entries = SkyImages.convert_catalog_to_celeste(cat_df, images);

# initialize tiled images and model parameters.  Don't fit the psf for now --
# we just need the tile_sources from mp.
tiled_blob, mp_all = ModelInit.initialize_celeste(images, cat_entries,
                                                  tile_width=20,
                                                  fit_psf=false);

#objid = "1237663784734359574" # Good
objid = "1237663784734359622" # Bad
s = findfirst(cat_df[:objid], objid)
cat_df[s, :]

relevant_sources = ModelInit.get_relevant_sources(mp_all, s);
cat_entries_s = cat_entries[relevant_sources];
tiled_blob, mp = ModelInit.initialize_celeste(images, cat_entries_s,
                                              tile_width=20,
                                              fit_psf=true);
active_s = findfirst(mp.objids, objid)
mp.active_sources = [ active_s ]

# TODO: This is slow but would run much faster if you had run
# limit_to_object_data() first.
trimmed_tiled_blob = ModelInit.trim_source_tiles(active_s, mp, tiled_blob;
                                                 noise_fraction=0.1);

using PyPlot
matshow(SkyImages.stitch_object_tiles(active_s, 3, mp, tiled_blob));
PyPlot.colorbar()


fit_time = time()
iter_count, max_f, max_x, result =
    OptimizeElbo.maximize_f(ElboDeriv.elbo, trimmed_tiled_blob, mp;
                            verbose=true, max_iters=50)
fit_time = time() - fit_time
