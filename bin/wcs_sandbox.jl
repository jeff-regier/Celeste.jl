using Celeste
using CelesteTypes

using DataFrames
using SampleData

import SDSS
import PSF
import FITSIO
import WCS
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
#objid = "1237662226208063491" # A bright star ... bad pixels though
objid = "1237662226208063565" # A brightish star but with good pixels.


#sub_rows_x = 1:150
#sub_rows_y = 1:150

width = 8

blob = deepcopy(original_blob);
reset_crpix!(blob);
cat_df = deepcopy(original_cat_df);
cat_loc = convert(Array{Float64}, cat_df[[:ra, :dec]]);
obj_row = original_cat_df[:objid] .== objid;
obj_loc = convert(Array, original_cat_df[obj_row, [:ra, :dec]])'[:]
#[ WCS.world_to_pixel(blob[b].wcs, obj_loc) for b=1:5]
entry_in_range = Bool[true for i=1:size(cat_loc, 1) ];
x_ranges = zeros(2, 5)
y_ranges = zeros(2, 5)
for b=1:5
	obj_loc_pix = WCS.world_to_pixel(blob[b].wcs, obj_loc)
	sub_rows_x = floor(obj_loc_pix[1] - width):ceil(obj_loc_pix[1] + width)
	sub_rows_y = floor(obj_loc_pix[2] - width):ceil(obj_loc_pix[2] + width)
	x_min = minimum(collect(sub_rows_x))
	y_min = minimum(collect(sub_rows_y))
	x_max = maximum(collect(sub_rows_x))
	y_max = maximum(collect(sub_rows_y))
	x_ranges[:, b] = Float64[x_min, x_max]
	y_ranges[:, b] = Float64[y_min, y_max]

	wcs_range = WCS.world_to_pixel(blob[b].wcs, cat_loc)
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
	blob[b].iota_vec = blob[b].iota_vec[x_min:x_max]
	blob[b].epsilon_mat = blob[b].epsilon_mat[x_min:x_max, y_min:y_max]
end
cat_df = cat_df[entry_in_range, :]
cat_entries = SDSS.convert_catalog_to_celeste(cat_df, blob);
cat_loc = convert(Array{Float64}, cat_df[[:ra, :dec]]);
initial_mp = ModelInit.cat_init(cat_entries, patch_radius=20.0, tile_width=5);

# Check the re-centering
for b=1:5
	println(WCS.world_to_pixel(blob[b].wcs, obj_loc))
	#println(WCS.world_to_pixel(blob[b].wcs, initial_mp.vp[1][ids.u]))
end

##############################
# Fit the image.

# Define a custom scaling.
custom_rect_rescaling = ones(length(UnconstrainedParams));
[custom_rect_rescaling[id] *= 1e-3 for id in ids_free.r1];
[custom_rect_rescaling[id] *= 1e5 for id in ids_free.u];
[custom_rect_rescaling[id] *= 1e1 for id in ids_free.a];

function custom_vp_to_rect!(vp::VariationalParams, vp_free::RectVariationalParams)
    Transform.vp_to_rect!(vp, vp_free, custom_rect_rescaling)
end

function custom_rect_to_vp!(vp_free::RectVariationalParams, vp::VariationalParams)
    Transform.rect_to_vp!(vp_free, vp, custom_rect_rescaling)
end

function custom_rect_unconstrain_sensitive_float(sf::SensitiveFloat, mp::ModelParams)
    Transform.rect_unconstrain_sensitive_float(sf, mp, custom_rect_rescaling)
end

custom_rect_transform = Transform.DataTransform(custom_rect_to_vp!, custom_vp_to_rect!,
                                     Transform.vector_to_free_vp!, Transform.free_vp_to_vector,
                                     custom_rect_unconstrain_sensitive_float,
                                     length(UnconstrainedParams));

# Some helper functions

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


# Set the psfs to the local psfs
fit_psfs = Array(Array{Float64, 2}, 5)
raw_psfs = Array(Array{Float64, 2}, 5)
psf_scales = Array(Float64, 5)
for b=1:5
	psf_point = WCS.world_to_pixel(blob[b].wcs, obj_loc) + Float64[ x_ranges[1, b], y_ranges[1, b]]
    raw_psf = PSF.get_psf_at_point(psf_point[1], psf_point[2], blob[b].raw_psf_comp);
    raw_psfs[b] = raw_psf / sum(raw_psf)
    psf_scales[b] = sum(raw_psf)
    psf_gmm, scale = PSF.fit_psf_gaussians(raw_psf, tol=1e-9, verbose=false);
    blob[b].psf = PSF.convert_gmm_to_celeste(psf_gmm, scale)
    fit_psfs[b] = PSF.get_psf_at_point(blob[b].psf)
end

this_star_fluxes = convert(Array{Float64}, cat_df[[:psfflux_u, :psfflux_g, :psfflux_r, :psfflux_i, :psfflux_z ]])[:]
synth_cat_entry = CatalogEntry(obj_loc, true, this_star_fluxes, this_star_fluxes, 0.1, .7, pi/4, 4.)
synth_blob = Synthetic.gen_blob(blob, [synth_cat_entry]; identity_wcs=false, expectation=true);

# Graph things in R since PyPlot is broken.
# Note that this star doesn't look a ton like the raw psf, especially in band 3.
for b=1:5
	writedlm("/tmp/raw_psf_$b.csv", raw_psfs[4], ',')
	writedlm("/tmp/fit_psf_$b.csv", fit_psfs[4], ',')
	writedlm("/tmp/pixels_$b.csv", blob[b].pixels, ',')
	writedlm("/tmp/synth_pixels_$b.csv", synth_blob[b].pixels, ',')
end

# The PSF is not great but it doesn't look so bad that it will
# completely destroy the ability to do inference.
nz = 16:35
println("==============================")
vcat(round(raw_psfs[b][nz, nz], 3),
     round(fit_psfs[b][nz, nz], 3))
round(1000. .* (fit_psfs[b][nz, nz] - raw_psfs[b][nz, nz]), 1)
println(psf_scales[b])


# Try varying background.
for b=1:5
	blob[b].constant_background = false
end
#include("src/ElboDeriv.jl"); include("src/OptimizeElbo.jl")
mp = deepcopy(initial_mp);
#res = OptimizeElbo.maximize_likelihood(blob, mp, Transform.rect_transform, xtol_rel=0);
res = OptimizeElbo.maximize_likelihood(blob, mp, custom_rect_transform);
compare_solutions(mp, initial_mp)
display_cat(cat_entries[1]);
get_brightness(mp)

# Try non-varying background.
for b=1:5
	blob[b].constant_background = true
end
#include("src/ElboDeriv.jl"); include("src/OptimizeElbo.jl")
mp_const = deepcopy(initial_mp);
#res = OptimizeElbo.maximize_elbo(blob, mp_const);
res = OptimizeElbo.maximize_likelihood(blob, mp_const, custom_rect_transform);
compare_solutions(mp, initial_mp)
display_cat(cat_entries[1]);
get_brightness(mp_const)


###################

b = 4
function get_e_g(img, mp)
	# Note: this is now broken due to needing the wcs Jacobian.

	ret = zero_sensitive_float(CanonicalParams, mp.S)
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
		println("Sources: $tile_sources    h,w range: $h_range $w_range")

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
		        wcs_jacobian = WCS.pixel_world_jacobian(tile.img.wcs, m_pos)'
		        for child_s in 1:length(tile_sources)
		            parent_s = tile_sources[child_s]
		            ElboDeriv.accum_pixel_source_stats!(sbs[parent_s], star_mcs, gal_mcs,
		                mp.vp[parent_s], child_s, parent_s, m_pos, tile.img.b,
		                fs0m, fs1m, E_G, var_G, wcs_jacobian)
		        end
		        e_image[h, w] = E_G.v * tile.img.iota
		        ElboDeriv.accum_pixel_ret!(tile_sources, this_pixel, tile.img.iota,
		            E_G, var_G, accum)
		    end
		end
	end

	e_image
end

e_images = [ get_e_g(blob[b], mp) for b=1:5 ];
for b=1:5
	writedlm("/tmp/e_image_$b.csv", e_images[b], ',')
end

H = blob[1].H
W = blob[2].W
[ round(e_images[b][i, j] - blob[b].pixels[i, j] / blob[b].iota, 1) for i=1:H, j=1:W, b=1:5 ]


###############
# Synthetic data
blob0 = deepcopy(blob);

function perturb_params(mp) # for testing derivatives != 0
    for vs in mp.vp
        vs[ids.a] = [ 0.4, 0.6 ]
        vs[ids.u[1]] += 1e-3
        vs[ids.u[2]] -= 1e-3
        vs[ids.r1] /= 10
        vs[ids.r2] *= 25.
        vs[ids.e_dev] += 0.05
        vs[ids.e_axis] += 0.05
        vs[ids.e_angle] += pi/10
        vs[ids.e_scale] *= 1.2
        vs[ids.c1] += 0.5
        vs[ids.c2] =  1e-1
    end
end

const sample_star_fluxes = [
    4.451805E+03,1.491065E+03,2.264545E+03,2.027004E+03,1.846822E+04]
const sample_galaxy_fluxes = [
    1.377666E+01, 5.635334E+01, 1.258656E+02,
    1.884264E+02, 2.351820E+02] * 100  # 1x wasn't bright enough

function sample_ce(pos, is_star::Bool)
    CatalogEntry(pos, is_star, sample_star_fluxes, sample_galaxy_fluxes,
        0.1, .7, pi/4, 4.)
end

one_body = [sample_ce(cat_entries[1].pos, true),];
synth_blob = Synthetic.gen_blob(blob0, one_body; identity_wcs=false);
for b=1:5
	synth_blob[b].constant_background = true
end
cat_mp = ModelInit.cat_init(one_body);
initial_mp = deepcopy(cat_mp);
perturb_params(initial_mp);
#get_e_g(synth_blob[3], initial_mp)
#synth_blob[3].pixels

mp = deepcopy(initial_mp);
#res = OptimizeElbo.maximize_elbo(synth_blob, mp);
res = OptimizeElbo.maximize_likelihood(synth_blob, mp);
compare_solutions(initial_mp, mp)
compare_solutions(cat_mp, mp)

get_brightness(mp)
sample_star_fluxes


#############################
# Plot our image.
pix_loc = WCS.world_to_pixel(blob[b].wcs, cat_loc)
WCS.pixel_to_world(blob[b].wcs, pix_loc)
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
	cat_px = WCS.world_to_pixel(blob[b].wcs, cat_loc)
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

in_poly = [ (x, y, WCS.point_within_radius_of_polygon(Float64[x, y], radius, poly))
            for x in -3:0.1:6, y in -3:0.1:6 ]
for p in in_poly
	if p[3]
		PyPlot.plot(p[1], p[2], "r+")
	end
end


##############
# Check the likelihood time

# Is this too slow?
@time WCS.pixel_deriv_to_world_deriv(original_blob[1].wcs, [1., 2.], [2., 4.])

mp = deepcopy(initial_mp);
lik_time = @time lik = ElboDeriv.elbo_likelihood(blob, mp);
@profile lik = ElboDeriv.elbo_likelihood(blob, mp);
Profile.print(format=:flat)
