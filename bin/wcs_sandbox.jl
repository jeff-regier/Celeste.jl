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


tile = ImageTile(5, 5, blob[1]);
# "Radius" is used in the sense of an L_{\infty} norm.
tile_width = 10
tr = tile_width / 2.  # tile radius
tc1 = tr + (tile.hh - 1) * tile_width
tc2 = tr + (tile.ww - 1) * tile_width

# Corners of the tile in pixel coordinates
tc = Float64[tr + (tile.hh - 1) * tile_width, tr + (tile.ww - 1) * tile_width]
tc11 = tc + Float64[-tr, -tr]
tc12 = tc + Float64[-tr, tr]
tc22 = tc + Float64[tr, tr]
tc21 = tc + Float64[tr, -tr]

# Convert the tile coordinates to a polygon in world coordinates.
tc_wcs = WCSLIB.wcsp2s(tile.img.wcs, hcat(tc11, tc12, tc22, tc21))'
PyPlot.plot(tc_wcs[:, 1], tc_wcs[:, 2])

for s in 1:mp.S
    pc = mp.patches[s].center  # patch center
    pr = mp.patches[s].radius  # patch radius

    if abs(pc[1] - tc_wcs[1]) <= (pr + tr) && abs(pc[2] - tc_wcs[2]) <= (pr + tr)
        push!(local_subset, s)
    end
end



# Try getting local sources
tile = ImageTile()