# Save an initialized celeste field in a JLD file.

using Celeste
using CelesteTypes
using SampleData
import JLD

field_dir = joinpath(dat_dir, "sample_field")
run_num = "003900"
camcol_num = "6"
field_num = "0269"

blob =
  SkyImages.load_sdss_blob(field_dir, run_num, camcol_num, field_num,
                        mask_planes=Set());

cat_df = SDSS.load_catalog_df(field_dir, run_num, camcol_num, field_num);
cat_entries = SkyImages.convert_catalog_to_celeste(cat_df, blob);

# TODO: change fit_psf to true when you're sure it works.
tiled_blob, mp_all =
  ModelInit.initialize_celeste(blob, cat_entries,
    tile_width=20, fit_psf=true);

# The blob cannot be saved because it has a pointer to a C++ object.
JLD.save("$dat_dir/initialzed_celeste_$(run_num)_$(camcol_num)_$(field_num).JLD",
         ["tiled_blob" => tiled_blob,
          "mp_all" => mp_all,
          "cat_df" => cat_df,
          "cat_entries" => cat_entries]);
