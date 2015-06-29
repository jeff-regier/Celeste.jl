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


##################
# Load a stamp to check out the psf and wcs

stamp_blob = SDSS.load_stamp_blob(dat_dir, "5.0073-0.0739");

#############
# Load and subsample the catalog

original_blob = SDSS.load_sdss_blob(field_dir, run_num, camcol_num, field_num);
original_cat_df = SDSS.load_catalog_df(field_dir, run_num, camcol_num, field_num);


##########################
# Select an object.

cat_loc = convert(Array{Float64}, original_cat_df[[:ra, :dec]]);
cat_pix = Util.world_to_pixel(original_blob[4].wcs, cat_loc)

obj_df = original_cat_df[[:objid, :is_star, :is_gal, :psfflux_r, :compflux_r]]
sort(obj_df[obj_df[:is_gal] .== true, :], cols=:compflux_r, rev=true)


#objid = "1237662226208063597" # A star that was obviously miscentered
#objid = "1237662226208063499" # A bright star
objid = "1237662226208063632" # A bright galaxy

obj_loc = cat_pix[original_cat_df[:objid] .== objid, :]

#sub_rows_x = 1:150
#sub_rows_y = 1:150

width = 20
sub_rows_x = floor(obj_loc[1] - width):ceil(obj_loc[1] + width)
sub_rows_y = floor(obj_loc[2] - width):ceil(obj_loc[2] + width)

x_min = minimum(collect(sub_rows_x))
y_min = minimum(collect(sub_rows_y))
x_max = maximum(collect(sub_rows_x))
y_max = maximum(collect(sub_rows_y))
blob = deepcopy(original_blob);
cat_df = deepcopy(original_cat_df);
cat_loc = convert(Array{Float64}, cat_df[[:ra, :dec]]);
entry_in_range = Bool[true for i=1:size(cat_loc, 1) ];
for b=1:5
	blob[b].pixels = blob[b].pixels[sub_rows_x, sub_rows_y]
	blob[b].H = size(blob[b].pixels, 1)
	blob[b].W = size(blob[b].pixels, 2)
	wcs_range = Util.world_to_pixel(blob[b].wcs, cat_loc)
	entry_in_range = entry_in_range &
		(x_min .<= wcs_range[:, 1] .<= x_max) &
		(y_min .<= wcs_range[:, 2] .<= y_max)
end
cat_df = cat_df[entry_in_range, :]
cat_entries = SDSS.convert_catalog_to_celeste(cat_df, blob);
cat_loc = convert(Array{Float64}, cat_df[[:ra, :dec]]);
initial_mp = ModelInit.cat_init(cat_entries, patch_radius=20.0, tile_width=5);


#############################
# Plot our image.
pix_loc = Util.world_to_pixel(blob[b].wcs, cat_loc)
Util.pixel_to_world(blob[b].wcs, pix_loc)
PyPlot.close("all")
for b=1:5
#for b=4
	# Plotting the transpose matches the images in the SDSS image browser
	# http://skyserver.sdss.org/dr7/en/tools/getimg/fields.asp

	# Here's how to look up an object:
	# http://skyserver.sdss.org/dr12/en/tools/explore/summary.aspx?id=1237662226208063597

	pixel_graph = blob[b].pixels
	clip = 8000
	pixel_graph[pixel_graph .>= clip] = clip
	PyPlot.figure()
	PyPlot.plt.subplot(1, 2, 1)
	PyPlot.imshow(pixel_graph', cmap=PyPlot.ColorMap("gray"), interpolation = "nearest")
	PyPlot.title("Band $b image\nObj $objid")

	PyPlot.plt.subplot(1, 2, 2)
	PyPlot.imshow(pixel_graph', cmap=PyPlot.ColorMap("gray"), interpolation = "nearest")
	# Expect that x_min and y_min are at least one and so take care of the PyPlot offset --
	# could write - (x_min - 1) - 1.
	cat_px = Util.world_to_pixel(blob[b].wcs, cat_loc)
	PyPlot.scatter(cat_px[:, 1] - x_min, cat_px[:, 2] - y_min, marker="o", c="r", s=25)

	obj_row = cat_df[:objid] .== objid 
	PyPlot.scatter(cat_px[obj_row, 1] - x_min, cat_px[obj_row, 2] - y_min,
		           marker="x", c="w", s=25)

	PyPlot.title("Band $b with catalog\nObj $objid")
end



##############################
# Fit the image.

function compare_solutions(mp1::ModelParams, mp2::ModelParams)
    # Compare the parameters, fits, and iterations.
    println("===================")
    println("Differences:")
    for var_name in names(ids)
        println(var_name)
        for s in 1:mp1.S
            println(s, ":\n", mp1.vp[s][ids.(var_name)], "\n", mp2.vp[s][ids.(var_name)])
        end
    end
    println("===================")
end

#include("src/ElboDeriv.jl"); include("src/OptimizeElbo.jl")
mp = deepcopy(initial_mp)
res = OptimizeElbo.maximize_elbo(blob, mp);
compare_solutions(mp, initial_mp)

function get_brightness(mp::ModelParams)
	brightness = [ElboDeriv.SourceBrightness(mp.vp[s]) for s in 1:mp.S];
	brightness_vals = [ Float64[b.E_l_a[i, j].v for i=1:size(b.E_l_a, 1), j=1:size(b.E_l_a, 2)] for b in brightness]
	brightness_vals
end



###################################
# Look at the psf.
psf_point_x = 80.
psf_point_y = 100.

raw_psf = Array(Array{Float64}, 5)
for b=1:5
	rrows, rnrow, rncol, cmat = SDSS.load_psf_data(field_dir, run_num, camcol_num, field_num, b);
	raw_psf[b] = PSF.get_psf_at_point(psf_point_x, psf_point_y, rrows, rnrow, rncol, cmat);
end

if false
	# This shows that imshow is screwed up and you have to offset the scatterplot by one.
	example_mat = zeros(11, 11)
	example_mat[6, 6] = 1.0
	PyPlot.figure()
	im = PyPlot.imshow(example_mat, interpolation = "nearest")
	PyPlot.scatter(6, 6, marker="o", c="r", s=25)
	PyPlot.scatter(5, 5, marker="o", c="k", s=25)

	b = 4
	PyPlot.figure()
	PyPlot.imshow(raw_psf[4], interpolation = "nearest")
	PyPlot.scatter(26 - 1, 26 - 1, marker="o", c="w", s=25)
	PyPlot.title("Band $b psf at ($psf_point_x, $psf_point_y)")
end


###################################
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




