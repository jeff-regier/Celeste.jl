# Run celeste interactively.

using Celeste

using DataFrames
using PyPlot

import FITSIO
import JLD
import ..SDSSIO
import ..WCSUtils
import ..SDSS

using Celeste.Types
import Celeste.SkyImages
import Celeste.ModelInit
import Celeste.OptimizeElbo
import Celeste.ElboDeriv

dir = joinpath(Pkg.dir("Celeste"), "test/data")

#http://skyserver.sdss.org/dr8/en/tools/explore/obj.asp?id=1237662226208063623
# 164.39678593,39.13884552

# run = 3900
# camcol = 6
# field = 269
#
# run = 3840
# camcol = 1
# field = 70

# objid = "1237662226208063623"

Logging.configure(level=Logging.DEBUG)

run = 4263
camcol = 5
field = 119

objid = "1237663784734491542"

make_cmd = "make RUN=$run CAMCOL=$camcol FIELD=$field"


images = SkyImages.read_sdss_field(run, camcol, field, dir);

# load catalog and convert to Array of `CatalogEntry`s.
run_string = @sprintf "%06d" run
camcol_string = @sprintf "%d" camcol
field_string = @sprintf "%04d" field

cat_df = Celeste.SDSS.load_catalog_df(dir, run_string, camcol_string, field_string);
cat_filename = @sprintf "photoObj-%06d-%d-%04d.fits" run camcol field
cat_entries = SkyImages.read_photoobj_celeste(joinpath(dir, cat_filename));

# initialize tiled images and model parameters.  Don't fit the psf for now --
# we just need the tile_sources from mp.
tiled_blob, mp = ModelInit.initialize_celeste(images, cat_entries,
                                              tile_width=20,
                                              fit_psf=false);
Celeste.WCSUtils.pix_to_world(images[3].wcs, [0., 0.])

# Choose an object:
s = findfirst(mp.objids, objid)
@assert s > 0
relevant_sources = ModelInit.get_relevant_sources(mp, s);
ModelInit.fit_object_psfs!(mp, relevant_sources, images);
mp.active_sources = [ s ];

# View
trimmed_tiled_blob =
  ModelInit.trim_source_tiles(s, mp, tiled_blob, noise_fraction=0.1);
band = 3
stitched_image, h_range, w_range =
  Celeste.SkyImages.stitch_object_tiles(s, band, mp, trimmed_tiled_blob, predicted=true);

using Celeste.WCSUtils
pix_loc = Celeste.WCSUtils.world_to_pix(
  mp.patches[s, band], mp.vp[s][ids.u])
matshow(stitched_image, vmax=1200);
PyPlot.plot(pix_loc[2] - w_range[1], pix_loc[1] - h_range[1], "wo", markersize=5)
PyPlot.colorbar()
PyPlot.title(objid)


fit_time = time()
iter_count, max_f, max_x, result =
    OptimizeElbo.maximize_f(ElboDeriv.elbo, trimmed_tiled_blob, mp;
                            verbose=true, max_iters=50)
fit_time = time() - fit_time
