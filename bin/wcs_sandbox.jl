using Celeste
using CelesteTypes

using DataFrames
using SampleData

import SDSS
import PSF
import FITSIO
#import PyPlot

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
original_crpix_band = Float64[unsafe_load(original_blob[b].wcs.crpix, i) for i=1:2, b=1:5];
function reset_crpix!(blob)
	for b=1:5
		unsafe_store!(blob[b].wcs.crpix, original_crpix_band[1, b], 1)
		unsafe_store!(blob[b].wcs.crpix, original_crpix_band[2, b], 2)
	end
end

cat_cols = [ :objid, :ra, :dec, :is_star, :frac_dev, :compflux_r, :psfflux_r ]
original_cat_df[:, cat_cols]

##########################
# Select an object.

cat_loc = convert(Array{Float64}, original_cat_df[[:ra, :dec]]);
obj_df = original_cat_df[[:objid, :is_star, :is_gal, :psfflux_r, :compflux_r]]
sort(obj_df[obj_df[:is_gal] .== true, :], cols=:compflux_r, rev=true)
sort(obj_df[obj_df[:is_gal] .== false, :], cols=:psfflux_r, rev=true)

#objid = "1237662226208063597" # A star that was obviously miscentered
#objid = "1237662226208063499" # A bright star also with lots of bad pixels
#objid = "1237662226208063632" # A bright galaxy
#objid = "1237662226208063541" # A bright star but with lots of bad pixels
#objid = "1237662226208063551" # A bright star but with lots of bad pixels
objid = "1237662226208063565" # A bright star


#sub_rows_x = 1:150
#sub_rows_y = 1:150

width = 8

blob = deepcopy(original_blob);
reset_crpix!(blob);
cat_df = deepcopy(original_cat_df);
cat_loc = convert(Array{Float64}, cat_df[[:ra, :dec]]);
obj_row = original_cat_df[:objid] .== objid;
obj_loc = convert(Array, original_cat_df[obj_row, [:ra, :dec]])'[:]
#[ Util.world_to_pixel(blob[b].wcs, obj_loc) for b=1:5]
entry_in_range = Bool[true for i=1:size(cat_loc, 1) ];
x_ranges = zeros(2, 5)
y_ranges = zeros(2, 5)
for b=1:5
	obj_loc_pix = Util.world_to_pixel(blob[b].wcs, obj_loc)
	sub_rows_x = floor(obj_loc_pix[1] - width):ceil(obj_loc_pix[1] + width)
	sub_rows_y = floor(obj_loc_pix[2] - width):ceil(obj_loc_pix[2] + width)
	x_min = minimum(collect(sub_rows_x))
	y_min = minimum(collect(sub_rows_y))
	x_max = maximum(collect(sub_rows_x))
	y_max = maximum(collect(sub_rows_y))
	x_ranges[:, b] = Float64[x_min, x_max]
	y_ranges[:, b] = Float64[y_min, y_max]

	wcs_range = Util.world_to_pixel(blob[b].wcs, cat_loc)
	entry_in_range = entry_in_range &
		(x_min .<= wcs_range[:, 1] .<= x_max) &
		(y_min .<= wcs_range[:, 2] .<= y_max)

	# Re-center the WCS coordinates
	crpix = original_crpix_band[:, b]
	unsafe_store!(blob[b].wcs.crpix, crpix[1] - x_min + 1, 1)
	unsafe_store!(blob[b].wcs.crpix, crpix[2] - y_min + 1, 2)
	
	blob[b].pixels = blob[b].pixels[sub_rows_x, sub_rows_y]
	blob[b].H = size(blob[b].pixels, 1)
	blob[b].W = size(blob[b].pixels, 2)
end
cat_df = cat_df[entry_in_range, :]
cat_entries = SDSS.convert_catalog_to_celeste(cat_df, blob);
cat_loc = convert(Array{Float64}, cat_df[[:ra, :dec]]);
initial_mp = ModelInit.cat_init(cat_entries, patch_radius=20.0, tile_width=5);

# Check the re-centering
for b=1:5
	println(Util.world_to_pixel(blob[b].wcs, obj_loc))
end

##############################
# Fit the image.

function compare_solutions(mp1::ModelParams, mp2::ModelParams)
    # Compare the parameters, fits, and iterations.
    println("===================")
    println("Differences:  mp1 vs mp2:")
    for var_name in names(ids)
        println(var_name)
        for s in 1:mp1.S
            println(s, ":\n", mp1.vp[s][ids.(var_name)], "\n", mp2.vp[s][ids.(var_name)])
        end
    end
    println("===================")
end

function get_brightness(mp::ModelParams)
	brightness = [ElboDeriv.SourceBrightness(mp.vp[s]) for s in 1:mp.S];
	brightness_vals = [ Float64[b.E_l_a[i, j].v for
		i=1:size(b.E_l_a, 1), j=1:size(b.E_l_a, 2)] for b in brightness]
	brightness_vals
end

function display_cat(cat_entry::CatalogEntry)
	[ println("$name: $(cat_entry.(name))") for name in names(cat_entry) ]
end

ids_free_names = Array(ASCIIString, length(ids_free))
for (name in names(ids_free)) 
	inds = ids_free.(name)
	for i = 1:length(inds)
		ids_free_names[inds[i]] = "$(name)_$(i)"
	end
end

ids_names = Array(ASCIIString, length(ids))
for (name in names(ids)) 
	inds = ids.(name)
	for i = 1:length(inds)
		ids_names[inds[i]] = "$(name)_$(i)"
	end
end


# Set the psfs to the local psfs
fit_psfs = Array(Array{Float64, 2}, 5)
raw_psfs = Array(Array{Float64, 2}, 5)
psf_scales = Array(Float64, 5)
for b=1:5
	psf_point = Util.world_to_pixel(blob[b].wcs, obj_loc)
    raw_psf = PSF.get_psf_at_point(psf_point[1], psf_point[2], blob[b].raw_psf_comp);
    raw_psfs[b] = raw_psf / sum(raw_psf)
    psf_scales[b] = sum(raw_psf)
    psf_gmm = PSF.fit_psf_gaussians(raw_psf, tol=1e-12, verbose=true);
    blob[b].psf = PSF.convert_gmm_to_celeste(psf_gmm)
    fit_psfs[b] = PSF.get_psf_at_point(blob[b].psf)
end

# The PSF is not great but it doesn't look so bad that it will
# completely destroy the ability to do inference.
nz = 16:35
println("==============================")
vcat(round(raw_psfs[b][nz, nz], 3),
     round(fit_psfs[b][nz, nz], 3))
round(1000. .* (fit_psfs[b][nz, nz] - raw_psfs[b][nz, nz]), 1)
println(psf_scales[b])



for b=1:5
	# Try varying background.
	blob[b].constant_background = false
end
#include("src/ElboDeriv.jl"); include("src/OptimizeElbo.jl")
mp = deepcopy(initial_mp);
mp.vp[1][ids.e_scale] = 0.5
#res = OptimizeElbo.maximize_likelihood(blob, mp, Transform.rect_transform, xtol_rel=0);
res = OptimizeElbo.maximize_likelihood(blob, mp);
# It says this star is a galaxy.
compare_solutions(mp, initial_mp)
# lik = ElboDeriv.elbo_likelihood(blob, mp);
# DataFrame(name=ids_names, d=lik.d[:,1])


# This gives pretty different values.
for b=1:5
	# Try non-varying background.
	blob[b].constant_background = true
end
#include("src/ElboDeriv.jl"); include("src/OptimizeElbo.jl")
mp_const = deepcopy(initial_mp);
#res = OptimizeElbo.maximize_elbo(blob, mp_const);
res = OptimizeElbo.maximize_likelihood(blob, mp_const);
compare_solutions(mp, mp_const)


# Look.
display_cat(cat_entries[1]);
get_brightness(mp)


###################

b = 4
function get_e_g(img, mp)
	ret = zero_sensitive_float(CanonicalParams, mp.S)
	img = blob[b]
	ElboDeriv.elbo_likelihood!(img, mp, ret)

	accum = ret
	accum.v += -sum(lfact(img.pixels[!isnan(img.pixels)]))

	star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(img.psf, mp, img.wcs)

	sbs = [ElboDeriv.SourceBrightness(mp.vp[s]) for s in 1:mp.S]

	WW = int(ceil(img.W / mp.tile_width))
	HH = int(ceil(img.H / mp.tile_width))
	e_image = zeros(img.H, img.W)
	for ww in 1:WW, hh in 1:HH
		tile = ImageTile(hh, ww, img)
		# might get a speedup from subsetting the mp here

		tile_sources = ElboDeriv.local_sources(tile, mp)
		h_range, w_range = ElboDeriv.tile_range(tile, mp.tile_width)
		println("$tile_sources $h_range $w_range")

		# fs0m and fs1m accumulate contributions from all sources
		fs0m = zero_sensitive_float(StarPosParams)
		fs1m = zero_sensitive_float(GalaxyPosParams)

		tile_S = length(tile_sources)
		E_G = zero_sensitive_float(CanonicalParams, tile_S)
		var_G = zero_sensitive_float(CanonicalParams, tile_S)

		# Iterate over pixels that are not NaN.
		for w in w_range, h in h_range
		    this_pixel = tile.img.pixels[h, w]
		    if !isnan(this_pixel)
		        clear!(E_G)
		        E_G.v = tile.img.epsilon
		        clear!(var_G)

		        m_pos = Float64[h, w]
		        wcs_jacobian = Util.pixel_world_jacobian(tile.img.wcs, m_pos)
		        for child_s in 1:length(tile_sources)
		            parent_s = tile_sources[child_s]
		            ElboDeriv.accum_pixel_source_stats!(sbs[parent_s], star_mcs, gal_mcs,
		                mp.vp[parent_s], child_s, parent_s, m_pos, tile.img.b,
		                fs0m, fs1m, E_G, var_G, wcs_jacobian)
		        end
		        println(E_G.v)
		        e_image[h, w] = E_G.v
		        ElboDeriv.accum_pixel_ret!(tile_sources, this_pixel, tile.img.iota,
		            E_G, var_G, accum)
		    end
		end
	end

	e_image
end

e_images = [ get_e_g(blob[b], mp) for b=1:5 ];

H = blob[1].H
W = blob[2].W
[ round(e_images[b][i, j] - blob[b].pixels[i, j] / blob[b].iota, 1) for i=1:H, j=1:W, b=1:5 ]




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


###################################
# Look at the psf.
psf_point_x = 80.
psf_point_y = 100.

raw_psf = Array(Array{Float64}, 5)
for b=1:5
	raw_psf_comp = SDSS.load_psf_data(field_dir, run_num, camcol_num, field_num, b);
	raw_psf[b] = PSF.get_psf_at_point(psf_point_x, psf_point_y, raw_psf_comp);
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


##############
# Check the likelihood time

# Is this too slow?
@time Util.pixel_deriv_to_world_deriv(original_blob[1].wcs, [1., 2.], [2., 4.])

mp = deepcopy(initial_mp);
lik_time = @time lik = ElboDeriv.elbo_likelihood(blob, mp);
@profile lik = ElboDeriv.elbo_likelihood(blob, mp);
Profile.print(format=:flat)
