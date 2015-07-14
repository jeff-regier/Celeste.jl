using Celeste
using CelesteTypes

using DataFrames
using SampleData

using ForwardDiff

import WCSLIB

zero_sensitive_float(CanonicalParams, Float64, 2)
zero_sensitive_float(CanonicalParams, ForwardDiff.Dual, 2)


loc = Float64[1. 1.; 0. 0.]
loc_dual = convert(Matrix{ForwardDiff.Dual}, loc)

WCSLIB.wcss2p(WCS.wcs_id, loc)
WCSLIB.wcss2p(WCS.wcs_id, loc_dual) # Fails
WCS.pixel_to_world(WCS.wcs_id, loc)
WCS.pixel_to_world(WCS.wcs_id, loc_dual)
















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
synth_blob = Synthetic.gen_blob(blob0, one_body);
for b=1:5
	synth_blob[b].constant_background = true
end
cat_mp = ModelInit.cat_init(one_body);
initial_mp = deepcopy(cat_mp);
perturb_params(initial_mp);
mp = deepcopy(initial_mp);


res = OptimizeElbo.maximize_likelihood(synth_blob, mp);
compare_solutions(initial_mp, mp)
compare_solutions(cat_mp, mp)

get_brightness(mp)
sample_star_fluxes


# Compare with forward differentiation
transform = custom_rect_transform;
omitted_ids = [ids_free.k[:], ids_free.c2[:], ids_free.r2];
x0 = transform.vp_to_vector(mp.vp, omitted_ids);
iter_count = 0

function objective(x)
    # Evaluate in the constrained space and then unconstrain again.
    transform.vector_to_vp!(x, mp.vp, omitted_ids)
    elbo = f(blob, mp)
    elbo.v
end

objective_grad = ForwardDiff.forwarddiff_gradient(objective, Float64, fadtype=:dual; n=length(x0));
g_fd = objective_grad(x0)