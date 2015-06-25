using Celeste
using CelesteTypes

using DataFrames
using SampleData

import SDSS
import PSF
import FITSIO
import PyPlot

# Some examples of the SDSS fits functions.
field_dir = joinpath(dat_dir, "sample_field")
run_num = "003900"
camcol_num = "6"
field_num = "0269"

const band_letters = ['u', 'g', 'r', 'i', 'z']

b = 1
b_letter = band_letters[b]

#############
# Load the catalog

blob = SDSS.load_sdss_blob(field_dir, run_num, camcol_num, field_num);
cat_df = SDSS.load_catalog_df(field_dir, run_num, camcol_num, field_num);
cat_entries = SDSS.convert_catalog_to_celeste(cat_df, blob);

coord = Array(Float64, 2, 1)
coord[1, 1] = 10.
coord[2, 1] = 20
WCSLIB.wcsp2s(blob[1].wcs, coord)


# To find the intersection of a circle with a quadrilateral.
p = Float64[0, 0]
v1 = Float64[1, -1]
v2 = Float64[1, 2]
r = Float64[-1, 0]

ray_crossing(p, r, v1, v2)




