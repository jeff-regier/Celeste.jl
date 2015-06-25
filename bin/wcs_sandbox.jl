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


# Plot neighboring points.
function make_rot_mat(theta::Float64)
    [ cos(theta) -sin(theta); sin(theta) cos(theta) ]
end

offset = [0, 0]
rot = make_rot_mat(pi / 3)
poly = Float64[1 1; -1 1; -1 -1; 1 -1]
poly = broadcast(+, poly, offset') * rot'
radius = 0.3

poly_graph = vcat(poly, poly[1,:])

PyPlot.plot(poly_graph[:,1],  poly_graph[:,2], "k")

in_poly = [ (x, y, Util.point_within_radius_of_polygon(Float64[x, y], radius, poly))
            for x in -3:0.1:6, y in -3:0.1:6 ]
for p in in_poly
	if p[3]
		PyPlot.plot(p[1], p[2], "r+")
	end
end


