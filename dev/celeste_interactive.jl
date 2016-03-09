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

function initialze_objid(
    objid::ASCIIString, mp_all::ModelParams{Float64},
    cat_entries::Array{Celeste.Types.CatalogEntry},
    images::Array{Celeste.Types.Image})

  s = findfirst(mp_all.objids, objid)
  #cat_df[s, :]

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
  trimmed_tiled_blob, mp, active_s, s
end


## Look at fluxes

MIN_FLUX = 1
flux_cols = [ symbol("psfflux_$b") for b in band_letters ]
min_fluxes = Float64[ minimum([ cat_df[row, flux_col] for flux_col in flux_cols ])
                      for row in 1:size(cat_df, 1) ];
max_fluxes = Float64[ maximum([ cat_df[row, flux_col] for flux_col in flux_cols ])
                      for row in 1:size(cat_df, 1) ];
#PyPlot.plt[:hist](max_fluxes, 200)
good_rows = max_fluxes .> 3;
sum(good_rows) / length(good_rows)

bad_objids = cat_df[:objid][!good_rows];

objid = "1237663784734359863" # Maybe a false rejection?
objid = "1237663784734359835" # Maybe a false rejection?

objid = "1237663784734359803" # An unnecessarily big field?
for objid in bad_objids
  trimmed_tiled_blob, mp, active_s, s = initialze_objid(objid, mp_all, cat_entries, images);
  band = 3
  stitched_image, h_range, w_range =
    SkyImages.stitch_object_tiles(active_s, band, mp, tiled_blob);

  pix_loc =
    WCSUtils.world_to_pix(mp.patches[active_s, band] , mp.vp[active_s][ids.u])
  matshow(stitched_image, vmax=1200);
  PyPlot.plot(pix_loc[2] - w_range[1] + 1, pix_loc[1] - h_range[1] + 1, "wo", markersize=5)
  PyPlot.colorbar()
  PyPlot.title(objid)
  PyPlot.savefig("/tmp/celeste_images/celeste_$objid.png")
  PyPlot.close()

end

# A borderline object:
objid = mp_all.objids[findmin(max_fluxes[good_rows])[2]]

#objid = "1237663784734359574" # Good
#objid = "1237663784734359622" # Bad


trimmed_tiled_blob, mp, active_s, s = initialze_objid(objid, mp_all, cat_entries, images);
cat_df[s, :]

fit_time = time()
iter_count, max_f, max_x, result =
    OptimizeElbo.maximize_f(ElboDeriv.elbo, trimmed_tiled_blob, mp;
                            verbose=true, max_iters=50)
fit_time = time() - fit_time
