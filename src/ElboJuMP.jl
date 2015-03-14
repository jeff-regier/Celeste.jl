using JuMP
using Celeste
using CelesteTypes
import Util
import SampleData

# Copy the parameters from the mp object into the JuMP model.
function SetJuMPParameters(mp::ModelParams)
	for s=1:mp.S
		setValue(vp_chi[s], mp.vp[s][ids.chi])
		setValue(vp_mu[s, 1], mp.vp[s][ids.mu][1])
		setValue(vp_mu[s, 2], mp.vp[s][ids.mu][2])

		setValue(vp_rho[s], mp.vp[s][ids.rho])
		setValue(vp_sigma[s], mp.vp[s][ids.sigma])
		setValue(vp_phi[s], mp.vp[s][ids.phi])

		setValue(vp_theta[s], mp.vp[s][ids.theta])
		for a=1:CelesteTypes.I
			setValue(vp_gamma[s, a], mp.vp[s][ids.gamma][a])
			setValue(vp_zeta[s, a], mp.vp[s][ids.zeta][a])
			for b=1:(CelesteTypes.B - 1)
				setValue(vp_beta[s, b, a], mp.vp[s][ids.beta][b, a])
				setValue(vp_lambda[s, b, a], mp.vp[s][ids.lambda][b, a])
			end
		end
	end
end


# Some simulated data.  blobs contains the image data, and
# mp is the parameter values.  three_bodies is not used.
# For now, treat these as global constants accessed within the expressions
blobs, mp, three_bodies = SampleData.gen_three_body_dataset();

max_height = 10
max_width = 10

# Reduce the size of the images for debugging
for b in 1:CelesteTypes.B
	this_height = min(blobs[b].H, max_height)
	this_width = min(blobs[b].W, max_width)
	blobs[b].H = this_height
	blobs[b].W = this_width
	blobs[b].pixels = blobs[b].pixels[1:this_width, 1:this_height] 
end

# Currently JuMP can't index into complex types, so convert everything to arrays.
blob_epsilon = [ blobs[img].epsilon for img=1:CelesteTypes.B ]

blob_pixels = [ blobs[img].pixels[ph, pw]
				for img=1:CelesteTypes.B, pw=1:img_w, ph=1:img_h ];
blob_iota = [ blobs[img].iota for img=1:CelesteTypes.B ]

# The number of gaussian components in the gaussian mixture representations
# of the PCF.
const n_pcf_comp = 3

# Below I use the fact that the number of colors is also the number
# of images in a blob.  TODO: change the indexing from b to img for clarity.

# These list comprehensions are necessary because JuMP can't index
# into immutable objects, it seems.
psf_xi_bar = [ blobs[b].psf[k].xiBar[row]
 		   for b=1:CelesteTypes.B, k=1:n_pcf_comp, row=1:2 ]
psf_sigma_bar = [ blobs[b].psf[k].SigmaBar[row, col]
                  for b=1:CelesteTypes.B, k=1:n_pcf_comp, row=1:2, col=1:2 ]
psf_alpha_bar = [ blobs[b].psf[k].alphaBar
                  for b=1:CelesteTypes.B, k=1:n_pcf_comp ]

# The number of normal components in the two galaxy types.
const n_gal1_comp = 8
const n_gal2_comp = 6

# Since there are different numbers of components per galaxy type,
# store them in different variables to avoid dealing with ragged arrays. 
galaxy_type1_sigma_tilde =
	[ galaxy_prototypes[1][g_k].sigmaTilde for g_k=1:n_gal1_comp ]
galaxy_type2_sigma_tilde =
	[ galaxy_prototypes[2][g_k].sigmaTilde for g_k=1:n_gal2_comp ]

galaxy_type1_alpha_tilde =
	[ galaxy_prototypes[1][g_k].alphaTilde for g_k=1:n_gal1_comp ]
galaxy_type2_alpha_tilde =
	[ galaxy_prototypes[2][g_k].alphaTilde for g_k=1:n_gal2_comp ]


# Make a data structure to allow JuMP to use only local sources.

# TOOD: Despite this maximum, I'm going to treat the rest of the code as if
# each image has the same number of pixels.
# Is it possible that these might be different for different
# images?  If so, it might be necessary to put some checks in the
# expressions below or handle this some other way.

img_w = maximum([ blobs[b].W for b=1:CelesteTypes.B ])
img_h = maximum([ blobs[b].H for b=1:CelesteTypes.B ])

# Get a rectangular array containing indicators of whether a
# source affects each pixel.
# TODO: is there a sparse multi-dimensional array object that JuMP can
# interact with?  Is there a better way to look inside ragged arrays
# in JuMP?
pixel_source_indicators = zeros(Int8, mp.S, img_w, img_h)

# NB: in the original code, pixel sources were tracked per image, but I
# don't see why that's necessary, so I'm just going to use one source
# list for all five bands.

# For now use Jeff's tile code with the first image.   This should be the
# same for each image.
img = blobs[1];
WW = int(ceil(img_w / mp.tile_width))
HH = int(ceil(img_h / mp.tile_width))
for ww in 1:WW, hh in 1:HH
    image_tile = ElboDeriv.ImageTile(hh, ww, img)
    this_local_sources = ElboDeriv.local_sources(image_tile, mp)
    h_range, w_range = ElboDeriv.tile_range(image_tile, mp.tile_width)
    for w in w_range, h in h_range, s in this_local_sources
    	pixel_source_indicators[s, w, h] = 1
    end
end


##########################
# Define the variational parameters.

celeste_m = Model()

# One set of variational parameters for each celestial object.
# These replace the ModelParams.vp object in the old version.

# The probability of being a galaxy.  (0 = star, 1 = galaxy)
@defVar(celeste_m, 0  <= vp_chi[s=1:mp.S] <= 1)

 # The location of the object.
@defVar(celeste_m, vp_mu[s=1:mp.S, axis=1:2])

# Ix1 scalar variational parameters for r_s.  The first
# row is for stars, and the second for galaxies (I think?).
@defVar(celeste_m, vp_gamma[s=1:mp.S, i=1:CelesteTypes.I] >= 0)
@defVar(celeste_m, vp_zeta[s=1:mp.S,  i=1:CelesteTypes.I] >= 0)

# The weight given to a galaxy of type 1.
@defVar(celeste_m, 0 <= vp_theta[s=1:mp.S] <= 1)

# galaxy minor/major ratio
@defVar(celeste_m, 0 <= vp_rho[s=1:mp.S] <= 1)

# galaxy angle
# TODO: bounds?
@defVar(celeste_m, vp_phi[s=1:mp.S])

# galaxy scale
@defVar(celeste_m, vp_sigma[s=1:mp.S] >= 0)

# The remaining parameters are matrices where the
# first column is for stars and the second is for galaxies.

# DxI matrix of color prior component indicators.
@defVar(celeste_m, 0 <= vp_kappa[s=1:mp.S, d=1:CelesteTypes.D, a=1:CelesteTypes.I] <= 1)

# (B - 1)xI matrices containing c_s means and variances, respectively.
@defVar(celeste_m, vp_beta[s=1:mp.S,   b=1:(CelesteTypes.B - 1), a=1:CelesteTypes.I])
@defVar(celeste_m, vp_lambda[s=1:mp.S, b=1:(CelesteTypes.B - 1), a=1:CelesteTypes.I] >= 0)

SetJuMPParameters(mp)


################
# Define the ELBO.  I consistently index objects with these names in this order:
# b / img: The color band or image that I'm looking at.  (Should be the same.)
# s: The source astronomical object
# k: The psf component
# g_k: The galaxy mixture component
# a: Colors only. Whether the variable is for a star or galaxy.
# pw: The w pixel value
# ph: The h pixel value
# *row, *col: Rows and columns of 2d vectors or matrices.  There is currently
# 	a bug in JuMP that requires these names not to be repeated, so I mostly
#   give objects different row and column names by prepending something to
#   "row" or "col".

########################
# Define the source brightness terms.

# Index 3 is r_s and  has a gamma expectation.
@defNLExpr(E_l_a_3[s=1:mp.S, a=1:CelesteTypes.I],
	       vp_gamma[s, a] * vp_zeta[s, a]);

# The remaining indices involve c_s and have lognormal
# expectations times E_c_3.
@defNLExpr(E_l_a_4[s=1:mp.S, a=1:CelesteTypes.I],
           E_l_a_3[s, a] * exp(vp_beta[s, 3, a] + .5 * vp_lambda[s, 3, a]));
@defNLExpr(E_l_a_5[s=1:mp.S, a=1:CelesteTypes.I],
           E_l_a_4[s, a] * exp(vp_beta[s, 4, a] + .5 * vp_lambda[s, 4, a]));

@defNLExpr(E_l_a_2[s=1:mp.S, a=1:CelesteTypes.I],
           E_l_a_3[s, a] * exp(-vp_beta[s, 2, a] + .5 * vp_lambda[s, 2, a]));
@defNLExpr(E_l_a_1[s=1:mp.S, a=1:CelesteTypes.I],
           E_l_a_2[s, a] * exp(-vp_beta[s, 1, a] + .5 * vp_lambda[s, 1, a]));

# Copy the brightnesses into a summable indexed structure.
@defNLExpr(E_l_a[s=1:mp.S, b=1:CelesteTypes.B, a=1:CelesteTypes.I],
	       (b == 1) * E_l_a_1[s, a] +
	       (b == 2) * E_l_a_2[s, a] +
	       (b == 3) * E_l_a_3[s, a] +
	       (b == 4) * E_l_a_4[s, a] +
	       (b == 5) * E_l_a_5[s, a]);

# Second order terms.
@defNLExpr(E_ll_a_3[s=1:mp.S, a=1:CelesteTypes.I],
	       vp_gamma[s, a] * (1 + vp_gamma[s, a]) * vp_zeta[s, a] ^ 2);

@defNLExpr(E_ll_a_4[s=1:mp.S, a=1:CelesteTypes.I],
	       E_ll_a_3[s, a] * exp(2 * vp_beta[s, 3, a] + 2 * vp_lambda[s, 3, a]));
@defNLExpr(E_ll_a_5[s=1:mp.S, a=1:CelesteTypes.I],
	       E_ll_a_4[s, a] * exp(2 * vp_beta[s, 4, a] + 2 * vp_lambda[s, 4, a]));

@defNLExpr(E_ll_a_2[s=1:mp.S, a=1:CelesteTypes.I],
	       E_ll_a_3[s, a] * exp(-2 * vp_beta[s, 2, a] + 2 * vp_lambda[s, 2, a]));
@defNLExpr(E_ll_a_1[s=1:mp.S, a=1:CelesteTypes.I],
	       E_ll_a_2[s, a] * exp(-2 * vp_beta[s, 1, a] + 2 * vp_lambda[s, 1, a]));

@defNLExpr(E_ll_a[s=1:mp.S, b=1:CelesteTypes.B, a=1:CelesteTypes.I],
	       (b == 1) * E_ll_a_1[s, a] +
	       (b == 2) * E_ll_a_2[s, a] +
	       (b == 3) * E_ll_a_3[s, a] +
	       (b == 4) * E_ll_a_4[s, a] +
	       (b == 5) * E_ll_a_5[s, a]);

####################################
# The bivariate normal mixtures, originally defined in load_bvn_mixtures

@defNLExpr(star_mean[b=1:CelesteTypes.B, s=1:mp.S, k=1:n_pcf_comp, row=1:2],
	       psf_xi_bar[b, k, row] + vp_mu[s, row]);

@defNLExpr(star_det[b=1:CelesteTypes.B, s=1:mp.S, k=1:n_pcf_comp],
	       psf_sigma_bar[b, k, 1, 1] * psf_sigma_bar[b, k, 2, 2] -
	       psf_sigma_bar[b, k, 1, 2] * psf_sigma_bar[b, k, 2, 1]);

# Matrix inversion by hand.  Maybe it would be better to have a super-variable
# indexing all bivariate normals so as not to repeat code, or to do some
# macro magic to build the same nonlinear expression for each inverse. 
@defNLExpr(star_precision[b=1:CelesteTypes.B,
	                      s=1:mp.S, k=1:n_pcf_comp, row=1:2, col=1:2],
           (sum{psf_sigma_bar[b, k, 2, 2]; row == 1 && col == 1} +
           	sum{psf_sigma_bar[b, k, 1, 1]; row == 2 && col == 2} -
           	sum{psf_sigma_bar[b, k, 1, 2]; row == 1 && col == 2} -
           	sum{psf_sigma_bar[b, k, 2, 1]; row == 2 && col == 1}) / star_det[b, s, k]);

@defNLExpr(star_z[b=1:CelesteTypes.B, s=1:mp.S, k=1:n_pcf_comp],
	       psf_alpha_bar[b, k] ./ (star_det[b, s, k] ^ 0.5 * 2pi));

#####################
# galaxy bvn components

# Terms originally from Util.get_bvn_cov(rho, phi, sigma):

# This is R
@defNLExpr(galaxy_rot_mat[s=1:mp.S, row=1:2, col=1:2],
		   (sum{ cos(vp_phi[s]); row == 1 && col == 1} +
			sum{-sin(vp_phi[s]); row == 1 && col == 2} +
			sum{ sin(vp_phi[s]); row == 2 && col == 1} +
			sum{ cos(vp_phi[s]); row == 2 && col == 2}));

# This is D
@defNLExpr(galaxy_scale_mat[s=1:mp.S, row=1:2, col=1:2],
			sum{1.0; row == 1 && col == 1} +
			sum{0.0; row == 1 && col == 2} +
			sum{0.0; row == 2 && col == 1} +
			sum{vp_rho[s]; row == 2 && col == 2});

# This is scale * D * R'.  Note that the column and row names
# circumvent what seems to be a bug in JuMP, see issue #415 in JuMP.jl
# on github.
@defNLExpr(galaxy_w_mat[s=1:mp.S, w_row=1:2, w_col=1:2],
		   vp_sigma[s] * sum{galaxy_scale_mat[s, w_row, sum_index] *
		                     galaxy_rot_mat[s, w_col, sum_index],
		                     sum_index = 1:2});

# This is W' * W
@defNLExpr(galaxy_xixi_mat[s=1:mp.S, xixi_row=1:2, xixi_col=1:2],
		   sum{galaxy_w_mat[s, xixi_sum_index, xixi_row] *
	           galaxy_w_mat[s, xixi_sum_index, xixi_col],
	           xixi_sum_index = 1:2});

# Terms from GalaxyCacheComponent:
# var_s and weight for type 1 galaxies:
@defNLExpr(galaxy_type1_var_s[b=1:CelesteTypes.B, s=1:mp.S,
	                          k=1:n_pcf_comp, g_k=1:n_gal1_comp,
	                          vars_row=1:2, vars_col=1:2],
	       psf_sigma_bar[b, k, vars_row, vars_col] +
	       galaxy_type1_sigma_tilde[g_k] * galaxy_xixi_mat[s, vars_row, vars_col]);

@defNLExpr(galaxy_type1_weight[b=1:CelesteTypes.B,
	                           k=1:n_pcf_comp, g_k=1:n_gal1_comp],
	       psf_alpha_bar[b, k] * galaxy_type1_alpha_tilde[g_k]);

# var_s and weight for type 2 galaxies:
@defNLExpr(galaxy_type2_var_s[b=1:CelesteTypes.B, s=1:mp.S,
	                          k=1:n_pcf_comp, g_k=1:n_gal2_comp,
	                          vars_row=1:2, vars_col=1:2],
	       psf_sigma_bar[b, k, vars_row, vars_col] +
	       galaxy_type2_sigma_tilde[g_k] * galaxy_xixi_mat[s, vars_row, vars_col]);

@defNLExpr(galaxy_type2_weight[b=1:CelesteTypes.B,
	                           k=1:n_pcf_comp, g_k=1:n_gal2_comp],
	       psf_alpha_bar[b, k] * galaxy_type2_alpha_tilde[g_k]);

# Now put these together to get the bivariate normal components,
# just like for the stars.

# The means are the same as for the stars.
# TODO: rename the mean so it's clear that the same quantity is being used for both.

# The determinant.  Note that the results were originally inaccurate without
# grouping the multiplication in parentheses, which is strange.  (This is no
# longer the case, maybe it was some weird artifact of the index name problem.)
@defNLExpr(galaxy_type1_det[b=1:CelesteTypes.B, s=1:mp.S,
	                        k=1:n_pcf_comp, g_k=1:n_gal1_comp],
	       (galaxy_type1_var_s[b, s, k, g_k, 1, 1] *
	        galaxy_type1_var_s[b, s, k, g_k, 2, 2]) -
	       (galaxy_type1_var_s[b, s, k, g_k, 1, 2] *
	        galaxy_type1_var_s[b, s, k, g_k, 2, 1]));

@defNLExpr(galaxy_type2_det[b=1:CelesteTypes.B, s=1:mp.S,
	                        k=1:n_pcf_comp, g_k=1:n_gal2_comp],
	       (galaxy_type2_var_s[b, s, k, g_k, 1, 1] *
	        galaxy_type2_var_s[b, s, k, g_k, 2, 2]) -
	       (galaxy_type2_var_s[b, s, k, g_k, 1, 2] *
	        galaxy_type2_var_s[b, s, k, g_k, 2, 1]));

# Matrix inversion by hand.  Also strangely, this is inaccurate if the
# minus signs are outside the sum.  (I haven't tested that since fixing the index
# name problem, so maybe that isn't true anymore either.)
@defNLExpr(galaxy_type1_precision[b=1:CelesteTypes.B, s=1:mp.S,
	                              k=1:n_pcf_comp, g_k=1:n_gal1_comp,
	                              prec_row=1:2, prec_col=1:2],
	       (sum{galaxy_type1_var_s[b, s, k, g_k, 2, 2];
	       	    prec_row == 1 && prec_col == 1} +
	       	sum{galaxy_type1_var_s[b, s, k, g_k, 1, 1];
	       	    prec_row == 2 && prec_col == 2} +
	       	sum{-galaxy_type1_var_s[b, s, k, g_k, 1, 2];
	       	    prec_row == 1 && prec_col == 2} +
	       	sum{-galaxy_type1_var_s[b, s, k, g_k, 2, 1];
	       	    prec_row == 2 && prec_col == 1}) /
	        galaxy_type1_det[b, s, k, g_k]);

@defNLExpr(galaxy_type2_precision[b=1:CelesteTypes.B, s=1:mp.S,
	                              k=1:n_pcf_comp, g_k=1:n_gal2_comp,
	                              prec_row=1:2, prec_col=1:2],
           (sum{galaxy_type2_var_s[b, s, k, g_k, 2, 2];
           	    prec_row == 1 && prec_col == 1} +
           	sum{galaxy_type2_var_s[b, s, k, g_k, 1, 1];
           	    prec_row == 2 && prec_col == 2} +
           	sum{-galaxy_type2_var_s[b, s, k, g_k, 1, 2];
           	    prec_row == 1 && prec_col == 2} +
           	sum{-galaxy_type2_var_s[b, s, k, g_k, 2, 1];
           	    prec_row == 2 && prec_col == 1}) /
           galaxy_type2_det[b, s, k, g_k]);

@defNLExpr(galaxy_type1_z[b=1:CelesteTypes.B, s=1:mp.S,
	                      k=1:n_pcf_comp, g_k=1:n_gal1_comp],
	        (galaxy_type1_alpha_tilde[g_k] * psf_alpha_bar[b, k]) ./
	        (galaxy_type1_det[b, s, k, g_k] ^ 0.5 * 2pi));

@defNLExpr(galaxy_type2_z[b=1:CelesteTypes.B, s=1:mp.S,
	                      k=1:n_pcf_comp, g_k=1:n_gal2_comp],
	       (galaxy_type2_alpha_tilde[g_k] * psf_alpha_bar[b, k]) ./
	       (galaxy_type2_det[b, s, k, g_k] ^ 0.5 * 2pi));


###########################
# Get the pdf values for each pixel.  Thie takes care of
# the functions accum_galaxy_pos and accum_star_pos.

# This allows us to have simpler expressions for the means.
# Note that this is in a perhaps counterintuitive order of
# h is "height" and w is "width", but I'll follow the convention in the
# original code.
@defNLExpr(pixel_locations[pw=1:img_w, ph=1:img_h, pixel_row=1:2],
	       sum{ph; pixel_row == 1} + sum{pw; pixel_row == 2});

# Reproduces
# function accum_star_pos!(bmc::BvnComponent,
#                          x::Vector{Float64},
#                          fs0m::SensitiveFloat)
# ... which called
# function ret_pdf(bmc::BvnComponent, x::Vector{Float64})
# TODO: This is the mean of both stars and galaxies, change the name to reflect this.
@defNLExpr(star_pdf_mean[img=1:CelesteTypes.B, s=1:mp.S, k=1:n_pcf_comp,
	                     pw=1:img_w, ph=1:img_h, pdf_mean_row=1:2],
           pixel_locations[pw, ph, pdf_mean_row] - star_mean[img, s, k, pdf_mean_row]);

@defNLExpr(star_pdf_f[img=1:CelesteTypes.B, s=1:mp.S, k=1:n_pcf_comp,
	                  pw=1:img_w, ph=1:img_h],
	        sum{
		        exp(-0.5 * sum{star_pdf_mean[img, s, k, pw, ph, pdf_f_row] * 
			        	       star_precision[img, s, k, pdf_f_row, pdf_f_col] *
			        	       star_pdf_mean[img, s, k, pw, ph, pdf_f_col],
			        	       pdf_f_row=1:2, pdf_f_col=1:2}) *
		        star_z[img, s, k];
		        pixel_source_indicators[s, pw, ph] == 1
		     });

# Galaxy pdfs
@defNLExpr(galaxy_type1_pdf_f[img=1:CelesteTypes.B, s=1:mp.S,
	                          k=1:n_pcf_comp, g_k=1:n_gal1_comp,
	                          pw=1:img_w, ph=1:img_h],
        sum{
	        exp(-0.5 * sum{star_pdf_mean[img, s, k, pw, ph, pdf_f_row] * 
		        	       galaxy_type1_precision[img, s, k, g_k, pdf_f_row, pdf_f_col] *
		        	       star_pdf_mean[img, s, k, pw, ph, pdf_f_col],
		        	       pdf_f_row=1:2, pdf_f_col=1:2}) *
	        galaxy_type1_z[img, s, k, g_k];
	        pixel_source_indicators[s, pw, ph] == 1
	     });

@defNLExpr(galaxy_type2_pdf_f[img=1:CelesteTypes.B, s=1:mp.S,
	                          k=1:n_pcf_comp, g_k=1:n_gal2_comp,
	                          pw=1:img_w, ph=1:img_h],
        sum{
	        exp(-0.5 * sum{star_pdf_mean[img, s, k, pw, ph, pdf_f_row] * 
		        	       galaxy_type2_precision[img, s, k, g_k, pdf_f_row, pdf_f_col] *
		        	       star_pdf_mean[img, s, k, pw, ph, pdf_f_col],
		        	       pdf_f_row=1:2, pdf_f_col=1:2}) *
	        galaxy_type2_z[img, s, k, g_k];
	        pixel_source_indicators[s, pw, ph] == 1
	     });

#############################
# Get the expectation and variance of G (accum_pixel_source_stats)

@defNLExpr(fs0m[img=1:CelesteTypes.B, s=1:mp.S, pw=1:img_w, ph=1:img_h],
	       sum{star_pdf_f[img, s, k, pw, ph], k=1:n_pcf_comp});

@defNLExpr(fs1m[img=1:CelesteTypes.B, s=1:mp.S, pw=1:img_w, ph=1:img_h],
	       sum{vp_theta[s] * galaxy_type1_pdf_f[img, s, k, g_k, pw, ph],
	           k=1:n_pcf_comp, g_k=1:n_gal1_comp} +
   	       sum{(1 - vp_theta[s]) * galaxy_type2_pdf_f[img, s, k, g_k, pw, ph],
	           k=1:n_pcf_comp, g_k=1:n_gal2_comp});

@defNLExpr(E_G[img=1:CelesteTypes.B, s=1:mp.S, pw=1:img_w, ph=1:img_h],
	       blob_epsilon[img] +
	       vp_chi[s] * E_l_a[s, img, 1] * fs0m[img, s, pw, ph] +
	       (1 - vp_chi[s]) * E_l_a[s, img, 2] * fs1m[img, s, pw, ph]);

@defNLExpr(Var_G[img=1:CelesteTypes.B, s=1:mp.S, pw=1:img_w, ph=1:img_h],
	       vp_chi[s] * E_ll_a[s, img, 1] * (fs0m[img, s, pw, ph] ^ 2) +
	       (1 - vp_chi[s]) *E_ll_a[s, img, 2] * (fs1m[img, s, pw, ph] ^ 2) -
	       E_G[img, s, pw, ph] ^ 2);

#####################
# Get the log likelihood (originally accum_pixel_ret)

# TODO: You could probably aggregate over images at this point, but I'll leave
# it like this for debugging.
@defNLExpr(img_log_likelihood[img=1:CelesteTypes.B],
	       sum{blob_pixels[img, pw, ph] *
	           (log(blob_iota[img]) +
	           	log(E_G[img, s, pw, ph]) -
	         	Var_G[img, s, pw, ph] / (2.0 * E_G[img, s, pw, ph] ^ 2)) -
	           blob_iota[img] * E_G[img, s, pw, ph],
	           pw=1:img_w, ph=1:img_h, s=1:mp.S});

@defNLExpr(elbo_log_likelihood,
	       sum{img_log_likelihood[img], img=1:CelesteTypes.B});
