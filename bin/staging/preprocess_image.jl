# Save an initialized celeste field in a JLD file.

using Celeste
using CelesteTypes
using SampleData
import JLD
import SloanDigitalSkySurvey: SDSS

# field_dir = joinpath(dat_dir, "sample_field")
# run_num = "003900"
# camcol_num = "6"
# field_num = "0269"

field_dir = joinpath(dat_dir, "stripe82_fields")
run_num = "000211"
camcol_num = "4"
field_num = "0227"

tile_width = 20

blob =
  SkyImages.load_sdss_blob(field_dir, run_num, camcol_num, field_num,
                        mask_planes=Set());

cat_df = SDSS.load_catalog_df(field_dir, run_num, camcol_num, field_num);
cat_entries = SkyImages.convert_catalog_to_celeste(cat_df, blob);

# TODO: Fitting all the psfs in advance takes an annoyingly long time,
# but initialize_celeste currently requires having the original blob, which
# cannot be saved in an HDF5 file because the wcs object is a C pointer.
# This could be refactored so that celeste can be re-initialized without the
# WCS allowing us to fit the psfs at optimization time rather than initialization..
tiled_blob, mp_all =
  ModelInit.initialize_celeste(blob, cat_entries,
    tile_width=tile_width, fit_psf=false);

# TODO: You could use limit_to_object_data() together with trim_source_tiles()
# to save only the data necessary for individual objects.

# The blob cannot be saved because it has a pointer to a C++ object.
JLD.save(string("$dat_dir/initialzed_celeste_",
                "$(run_num)_$(camcol_num)_$(field_num)_$(tile_width)px.jld"),
         Dict("tiled_blob" => tiled_blob,
              "mp_all" => mp_all,
              "cat_df" => cat_df,
              "cat_entries" => cat_entries));
