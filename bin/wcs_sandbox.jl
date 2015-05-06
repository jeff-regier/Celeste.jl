using Celeste
using CelesteTypes

using FITSIO
using WCSLIB
using DataFrames
using SampleData


field_dir = joinpath(dat_dir, "sample_field")
run_num = "003900"
camcol_num = "6"
frame_num = "0269"

this_field = SDSS.load_field(field_dir, run_num, camcol_num, frame_num);

# Doesn't work:
this_cat = SDSS.load_catalog(field_dir, run_num, camcol_num, frame_num);


stamp_id = "164.4311-39.0359"
blob = SDSS.load_stamp_blob(dat_dir, stamp_id);
cat = SDSS.load_stamp_catalog(dat_dir, stamp_id, blob);