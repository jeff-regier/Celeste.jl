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
# Load and subsample the catalog

original_blob = SDSS.load_sdss_blob(field_dir, run_num, camcol_num, field_num);
original_cat_df = SDSS.load_catalog_df(field_dir, run_num, camcol_num, field_num);

blob = deepcopy(original_blob);
cat_df = deepcopy(original_cat_df);

sub_rows = 200:300
min_row = minimum(collect(sub_rows))
max_row = maximum(collect(sub_rows))
cat_loc = convert(Array{Float64}, cat_df[[:ra, :dec]])';
entry_in_range = Bool[true for i=1:size(cat_loc, 2) ];
for b=1:5
	blob[b].pixels = blob[b].pixels[sub_rows, sub_rows]
	blob[b].H = size(blob[b].pixels, 1)
	blob[b].W = size(blob[b].pixels, 2)
	wcs_range = WCSLIB.wcss2p(blob[b].wcs, cat_loc)'
	entry_in_range = entry_in_range &
		(min_row .<= wcs_range[:, 1] .<= max_row) &
		(min_row .<= wcs_range[:, 2] .<= max_row)
end
cat_df = cat_df[entry_in_range, :]
cat_entries = SDSS.convert_catalog_to_celeste(cat_df, blob);
cat_loc = convert(Array{Float64}, cat_df[[:ra, :dec]])';


b = 3
PyPlot.close()
PyPlot.plt.subplot(1, 2, 1)
PyPlot.imshow(blob[b].pixels, cmap=PyPlot.ColorMap("gray"))

PyPlot.plt.subplot(1, 2, 2)
PyPlot.imshow(blob[b].pixels, cmap=PyPlot.ColorMap("gray"))
cat_px = WCSLIB.wcss2p(blob[b].wcs, cat_loc)' - min_row
PyPlot.scatter(cat_px[:, 2], cat_px[:, 1], marker="o", c="r", s=50)





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




