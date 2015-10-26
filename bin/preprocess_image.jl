# This script processes a real image and saves it as a JLD file.
using Celeste
using CelesteTypes
using SampleData

# An actual celestial body.
field_dir = joinpath(dat_dir, "sample_field")
run_num = "003900"
camcol_num = "6"
field_num = "0269"

original_blob =
  SkyImages.load_sdss_blob(field_dir, run_num, camcol_num, field_num,
                        mask_planes=Set());

original_cat_df =
  SDSS.load_catalog_df(field_dir, run_num, camcol_num, field_num);
original_cat_entries =
  SkyImages.convert_catalog_to_celeste(original_cat_df, original_blob);

tiled_blob, mp_original_all =
  ModelInit.initialize_celeste(
    original_blob, original_cat_entries,
    tile_width=tile_width, fit_psf=false);
