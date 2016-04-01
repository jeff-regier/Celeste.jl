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

# objid = "1237663784734490878" # too bright
objid = "1237663784734490990" # Bad angle

make_cmd = "make RUN=$run CAMCOL=$camcol FIELD=$field"

images = SkyImages.read_sdss_field(run, camcol, field, dir);

# load catalog and convert to Array of `CatalogEntry`s.
run_string = @sprintf "%06d" run
camcol_string = @sprintf "%d" camcol
field_string = @sprintf "%04d" field

cat_df = Celeste.SDSS.load_catalog_df(dir, run_string, camcol_string, field_string);
cat_filename = @sprintf "photoObj-%06d-%d-%04d.fits" run camcol field
#cat_entries = SkyImages.read_photoobj_celeste(joinpath(dir, cat_filename));

catalog = Celeste.SDSSIO.read_photoobj(joinpath(dir, cat_filename));
cat_entries = Vector{CatalogEntry}(catalog);

# initialize tiled images and model parameters.  Don't fit the psf for now --
# we just need the tile_sources from mp.
tiled_blob, mp_all =
  ModelInit.initialize_celeste(images, cat_entries,
                               tile_width=20, fit_psf=false);

# Choose an object:
s_obj = findfirst(mp_all.objids, objid)
@assert s_obj > 0
cat_entries[s_obj]
mp_all.active_sources = [ s_obj ];
relevant_sources = ModelInit.get_relevant_sources(mp_all, s_obj);
mp = ModelParams(mp_all, relevant_sources);
ModelInit.fit_object_psfs!(mp, collect(1:mp.S), images);
s = mp.active_sources[1]

cat_entries[s_obj].gal_angle * 180 / pi
mp_all.vp[s_obj][ids.e_angle] * 180 / pi

# View
trimmed_tiled_blob =
  ModelInit.trim_source_tiles(s, mp, tiled_blob, noise_fraction=0.1);

image_bands = 1:5

function plot_object(
    trimmed_tiled_blob, mp, s, image_bands;
    predicted::Bool=true, title::ASCIIString="", vmax=NaN)

  PyPlot.figure()
  PyPlot.title("$objid")
  stitched_image_vec = Array(Matrix{Float64}, length(image_bands))
  for band_ind = 1:length(image_bands)
    band = image_bands[band_ind]
    stitched_image, h_range, w_range = Celeste.SkyImages.stitch_object_tiles(
      s, band, mp, trimmed_tiled_blob, predicted=predicted);
    pix_loc = Celeste.WCSUtils.world_to_pix(
      mp.patches[s, band], mp.vp[s][ids.u])

    PyPlot.subplot(1, length(image_bands), band_ind)
    if isnan(vmax)
      PyPlot.imshow(stitched_image, interpolation="None");
    else
      PyPlot.imshow(stitched_image, interpolation="None", vmax=vmax);
    end
    PyPlot.plot(pix_loc[2] - w_range[1], pix_loc[1] - h_range[1], "wo", markersize=5)
    PyPlot.colorbar()
    PyPlot.title("band $band $title")
    stitched_image_vec[band_ind] = stitched_image
  end

  stitched_image_vec
end


#omitted_ids = Int[]
simplex_min = 0.006
mp.vp[s][ids.a] = [ simplex_min, 1.0 - simplex_min ]
omitted_ids = ids_free.a

fit_time = time()
iter_count, max_f, max_x, result =
    OptimizeElbo.maximize_f(ElboDeriv.elbo, trimmed_tiled_blob, mp;
                            verbose=true, max_iters=200, omitted_ids=omitted_ids)
fit_time = time() - fit_time

brightness = ElboDeriv.get_brightness(mp)[1]
ElboDeriv.get_brightness(mp_all)[1]

PyPlot.close("all")
band = 3
vmax = NaN


cat_img = plot_object(trimmed_tiled_blob, mp_all, s_obj, Int[band], predicted=true, title="catalog", vmax=vmax);
fit_img = plot_object(trimmed_tiled_blob, mp, s, Int[band], predicted=true, title="fit", vmax=vmax);
orig_img = plot_object(trimmed_tiled_blob, mp, s, Int[band], predicted=false, title="image", vmax=vmax);

mp_test = deepcopy(mp);
mp_test.vp[s][ids.e_angle] = 180 * pi / 180
mp_test.vp[s][ids.e_axis] = 0.1
mp_test.vp[s][ids.e_scale] = 10
plot_object(trimmed_tiled_blob, mp_test, s, Int[band], predicted=true, title="image", vmax=vmax);
PyPlot.title(mp_test.vp[s][ids.e_angle] * 180 / pi)


fit_residual = fit_img[1] - orig_img[1];
cat_residual = cat_img[1] - orig_img[1];
sum(abs(fit_residual[!isnan(fit_residual)]))
sum(abs(cat_residual[!isnan(cat_residual)]))
@assert sum(!isnan(cat_residual)) == sum(!isnan(fit_residual))

vmax = max(maximum(abs(fit_residual)), maximum(abs(cat_residual)))
PyPlot.matshow(abs(fit_residual), vmax=vmax, vmin=-vmax)
PyPlot.title("fit residual")
PyPlot.colorbar()

PyPlot.matshow(abs(cat_img[1] - orig_img[1]), vmax=vmax, vmin=-vmax)
PyPlot.title("cat residual")
PyPlot.colorbar()
