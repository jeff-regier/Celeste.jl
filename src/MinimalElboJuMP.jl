# This file implements a minimal version of the ELBO log likelihood in JuMP.
# For the time being, 
# use it by including it in a namespace that has the following objects:
#
# blobs:  An array of Image objects (of length CelestTypes.B)
# mp:     A ModelParams object
# E_l_a:  An array of brightness first moments (see ElboDeriv.jl)
# E_ll_a: An array of brightness second moments (see ElboDeriv.jl)
#
# To include the file, use:
# include(joinpath(Pkg.dir("Celeste"), "src/ElboJuMP.jl"))


using JuMP
using Celeste
using CelesteTypes
import Util
import SampleData

# Copy the parameters from the mp object into the JuMP model.
function SetJuMPParameters(mp::ModelParams)
	for s=1:mp.S
		setValue(vp_mu[s, 1], mp.vp[s][ids.mu][1])
		setValue(vp_mu[s, 2], mp.vp[s][ids.mu][2])
	end
end

#########################
# Define some global constants related to the problem.

# The number of gaussian components in the gaussian mixture representations
# of the PCF.
const n_pcf_comp = 3

#####################
# First some code to convert the original data structures to
# multidimensional arrays that can be accessed in JuMP.

# TOOD: Despite this maximum, I'm going to treat the rest of the code as if
# each image has the same number of pixels.
# Is it possible that these might be different for different
# images?  If so, it might be necessary to put some checks in the
# expressions below or handle this some other way.

img_w = maximum([ blobs[b].W for b=1:CelesteTypes.B ])
img_h = maximum([ blobs[b].H for b=1:CelesteTypes.B ])

# Currently JuMP can't index into complex types, so convert everything to arrays.
blob_epsilon = [ blobs[img].epsilon for img=1:CelesteTypes.B ]

blob_pixels = [ blobs[img].pixels[ph, pw]
				for img=1:CelesteTypes.B, pw=1:img_w, ph=1:img_h ];
blob_iota = [ blobs[img].iota for img=1:CelesteTypes.B ]

# Below I use the fact that the number of colors is also the number
# of images in a blob.  TODO: change the indexing from b to img for clarity.

psf_xi_bar = [ blobs[b].psf[k].xiBar[row]
 		   for b=1:CelesteTypes.B, k=1:n_pcf_comp, row=1:2 ]
psf_sigma_bar = [ blobs[b].psf[k].SigmaBar[row, col]
                  for b=1:CelesteTypes.B, k=1:n_pcf_comp, row=1:2, col=1:2 ]
psf_alpha_bar = [ blobs[b].psf[k].alphaBar
                  for b=1:CelesteTypes.B, k=1:n_pcf_comp ]

# The constant contribution to the log likelihood of the x! terms.
log_base_measure = [ -sum(lfact(blobs[b].pixels)) for b=1:CelesteTypes.B ] 

# Not every source affects every pixel.  Encode which sources affect
# which pixels in an rectangular array of zeros and ones.
# TODO: is there a sparse multi-dimensional array object that JuMP can
# interact with?  Is there a better way to look inside ragged arrays
# in JuMP?
pixel_source_indicators = zeros(Int8, mp.S, img_w, img_h);

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
pixel_source_count = [ sum(pixel_source_indicators[:, pw, ph]) for pw=1:img_w, ph=1:img_h];


##########################
# Define the variational parameters.

celeste_m = Model()

# One set of variational parameters for each celestial object.
# These replace the ModelParams.vp object in the old version.

 # The location of the object.
@defVar(celeste_m, vp_mu[s=1:mp.S, axis=1:2])

SetJuMPParameters(mp)

################
# Define the ELBO.  I consistently index objects with these names in this order:
# b / img: The color band or image that I'm looking at.  (Should be the same.)
# s: The source astronomical object
# k: The psf component
# pw: The w pixel value
# ph: The h pixel value
# *row, *col: Rows and columns of 2d vectors or matrices.  There is currently
# 	a bug in JuMP that requires these names not to be repeated, so I mostly
#   give objects different row and column names by prepending something to
#   "row" or "col".

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

#############################
# Get the expectation and variance of G (accum_pixel_source_stats)

# Star density values:
@defNLExpr(fs0m[img=1:CelesteTypes.B, s=1:mp.S, pw=1:img_w, ph=1:img_h],
	       sum{star_pdf_f[img, s, k, pw, ph], k=1:n_pcf_comp});

@defNLExpr(E_G_s[img=1:CelesteTypes.B, s=1:mp.S, pw=1:img_w, ph=1:img_h],
	       E_l_a[s, img, 1] * fs0m[img, s, pw, ph]);

@defNLExpr(Var_G_s[img=1:CelesteTypes.B, s=1:mp.S, pw=1:img_w, ph=1:img_h],
	       E_ll_a[s, img, 1] * (fs0m[img, s, pw, ph] ^ 2) -
	       (E_G_s[img, s, pw, ph] ^ 2));

@defNLExpr(E_G[img=1:CelesteTypes.B, pw=1:img_w, ph=1:img_h],
	       sum{E_G_s[img, s, pw, ph], s=1:mp.S} + blob_epsilon[img]);

@defNLExpr(Var_G[img=1:CelesteTypes.B, pw=1:img_w, ph=1:img_h],
	       sum{Var_G_s[img, s, pw, ph], s=1:mp.S});

#####################
# Get the log likelihood (originally accum_pixel_ret)

# TODO: Use pixel_source_count to not use the delta-method approximation
# when there are no sources in a pixel.
@defNLExpr(pixel_log_likelihood[img=1:CelesteTypes.B, pw=1:img_w, ph=1:img_h],
	       blob_pixels[img, pw, ph] *
	        (log(blob_iota[img]) +
	         log(E_G[img, pw, ph]) -
	         Var_G[img, pw, ph] / (2.0 * (E_G[img, pw, ph] ^ 2))) -
	        blob_iota[img] * E_G[img, pw, ph]);

@defNLExpr(img_log_likelihood[img=1:CelesteTypes.B],
	       sum{pixel_log_likelihood[img, pw, ph],
	           pw=1:img_w, ph=1:img_h});

@defNLExpr(elbo_log_likelihood,
	       sum{img_log_likelihood[img] + log_base_measure[img],
	       img=1:CelesteTypes.B});

# Don't print the last expression when including!
1