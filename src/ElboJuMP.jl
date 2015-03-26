# This file implements the ELBO log likelihood in JuMP.


module ElboJuMP

using JuMP
using Celeste
using CelesteTypes
import Util
import SampleData
import ElboDeriv


# The number of gaussian components in the gaussian mixture representations
# of the PCF.
const n_pcf_comp = 3

# The number of normal components in the two galaxy types.
const n_gal1_comp = 8
const n_gal2_comp = 6

using ReverseDiffSparse
eval(ReverseDiffSparse, :(const SPLAT_THRESHOLD = 1000))


function build_jump_model(blob::Blob, mp::ModelParams)
    # First some code to convert the original data structures to
    # multidimensional arrays that can be accessed in JuMP.

    # Despite this maximum, I'm going to treat the rest of the code as if
    # each image has the same number of pixels.
    # Is it possible that these might be different for different
    # images?  If so, it might be necessary to put some checks in the
    # expressions below or handle this some other way.

    img_w = maximum([ blob[b].W for b=1:CelesteTypes.B ])
    img_h = maximum([ blob[b].H for b=1:CelesteTypes.B ])

    # Currently JuMP can't index into complex types, so convert everything to arrays.
    blob_epsilon = [ blob[img].epsilon for img=1:CelesteTypes.B ]

    blob_pixels = [ blob[img].pixels[ph, pw]
                    for img=1:CelesteTypes.B, pw=1:img_w, ph=1:img_h ];
    blob_iota = [ blob[img].iota for img=1:CelesteTypes.B ]

    # Below I use the fact that the number of colors is also the number
    # of images in a blob.  TODO: change the indexing from b to img for clarity.

    # These list comprehensions are necessary because JuMP can't index
    # into immutable objects, it seems.
    psf_xi_bar = [ blob[b].psf[k].xiBar[row]
               for b=1:CelesteTypes.B, k=1:n_pcf_comp, row=1:2 ]
    psf_sigma_bar = [ blob[b].psf[k].SigmaBar[row, col]
                      for b=1:CelesteTypes.B, k=1:n_pcf_comp, row=1:2, col=1:2 ]
    psf_alpha_bar = [ blob[b].psf[k].alphaBar
                      for b=1:CelesteTypes.B, k=1:n_pcf_comp ]

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

    # The constant contribution to the log likelihood of the x! terms.
    log_base_measure = [ -sum(lfact(blob[b].pixels)) for b=1:CelesteTypes.B ] 

    # This allows us to have simpler expressions for the means.
    # Note that this is in a perhaps counterintuitive order of
    # h is "height" and w is "width", but I'll follow the convention in the
    # original code.
    pixel_locations = Int64[ (pixel_row == 1) * ph + (pixel_row == 2) * pw
                         for pw=1:img_w, ph=1:img_h, pixel_row=1:2 ];

    # NB: in the original code, pixel sources were tracked per image, but I
    # don't see why that's necessary, so I'm just going to use one source
    # list for all five bands.

    # For now use Jeff's tile code with the first image.   This should be the
    # same for each image.
    img = blob[1];
    WW = convert(Int, ceil(img_w / mp.tile_width))
    HH = convert(Int, ceil(img_h / mp.tile_width))

    # An array ocontainingwhich pixels are associated with each source.
    source_pixels = Array(Vector{(Int64, Int64)}, mp.S);
    for s = 1:mp.S
        source_pixels[s] = Array((Int64, Int64), 0)
    end

    # An array contining which sources are associated with
    # each pixel.
    pixel_sources = Array(Vector{Int64}, img_w, img_h);
    for pw=1:img_w, ph=1:img_h
        pixel_sources[pw, ph] = Array(Int64, 0)
    end

    for ww in 1:WW, hh in 1:HH
        image_tile = ElboDeriv.ImageTile(hh, ww, img)
        this_local_sources = ElboDeriv.local_sources(image_tile, mp)
        h_range, w_range = ElboDeriv.tile_range(image_tile, mp.tile_width)
        for w in w_range, h in h_range, s in this_local_sources
            push!(source_pixels[s], (w, h))
            push!(pixel_sources[w, h], s)
        end
    end

    # Define the variational parameters.

    jump_m = Model()

    # One set of variational parameters for each celestial object.
    # These replace the ModelParams.vp object in the old version.

    # The probability of being a galaxy.  (0 = star, 1 = galaxy)
    @defVar(jump_m, 0  <= vp_chi[s=1:mp.S] <= 1)

     # The location of the object.
    @defVar(jump_m, vp_mu[s=1:mp.S, axis=1:2])

    # Ix1 scalar variational parameters for r_s.  The first
    # row is for stars, and the second for galaxies (I think?).
    @defVar(jump_m, vp_gamma[s=1:mp.S, i=1:CelesteTypes.I] >= 0)
    @defVar(jump_m, vp_zeta[s=1:mp.S,  i=1:CelesteTypes.I] >= 0)

    # The weight given to a galaxy of type 1.
    @defVar(jump_m, 0 <= vp_theta[s=1:mp.S] <= 1)

    # galaxy minor/major ratio
    @defVar(jump_m, 0 <= vp_rho[s=1:mp.S] <= 1)

    # galaxy angle
    # TODO: bounds?
    @defVar(jump_m, vp_phi[s=1:mp.S])

    # galaxy scale
    @defVar(jump_m, vp_sigma[s=1:mp.S] >= 0)

    # The remaining parameters are matrices where the
    # first column is for stars and the second is for galaxies.

    # DxI matrix of color prior component indicators.
    @defVar(jump_m, 0 <= vp_kappa[s=1:mp.S, d=1:CelesteTypes.D, a=1:CelesteTypes.I] <= 1)

    # (B - 1)xI matrices containing c_s means and variances, respectively.
    @defVar(jump_m, vp_beta[s=1:mp.S,   b=1:(CelesteTypes.B - 1), a=1:CelesteTypes.I])
    @defVar(jump_m, vp_lambda[s=1:mp.S, b=1:(CelesteTypes.B - 1), a=1:CelesteTypes.I] >= 0)

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

    # Define the ELBO.  I (almost) consistently index objects with these names in this order:
    # b / img: The color band or image that I'm looking at.  (Should be the same.)
    #          Note: the brightness objects currently reverse the order of b and s.
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

    # Copy the brightnesses into an easily indexed structure.
    # TODO: make the indexing consistent with the rest of the file by
    # putting the band first.
    @defNLExpr(E_l_a[s=1:mp.S, b=1:CelesteTypes.B, a=1:CelesteTypes.I],
               sum{E_l_a_1[s, a]; b == 1} +
               sum{E_l_a_2[s, a]; b == 2} +
               sum{E_l_a_3[s, a]; b == 3} +
               sum{E_l_a_4[s, a]; b == 4} +
               sum{E_l_a_5[s, a]; b == 5});

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

    # TODO: make the indexing consistent with the rest of the file by
    # putting the band first.
    @defNLExpr(E_ll_a[s=1:mp.S, b=1:CelesteTypes.B, a=1:CelesteTypes.I],
               sum{ E_ll_a_1[s, a]; b == 1 } +
               sum{ E_ll_a_2[s, a]; b == 2 } +
               sum{ E_ll_a_3[s, a]; b == 3 } +
               sum{ E_ll_a_4[s, a]; b == 4 } +
               sum{ E_ll_a_5[s, a]; b == 5 });

    ####################################
    # The bivariate normal mixtures, originally defined in load_bvn_mixtures

    @defNLExpr(star_mean[b=1:CelesteTypes.B, s=1:mp.S, k=1:n_pcf_comp, row=1:2],
               psf_xi_bar[b, k, row] + vp_mu[s, row]);

    # Matrix inversion by hand.
    star_det = Float64[ (psf_sigma_bar[b, k, 1, 1] * psf_sigma_bar[b, k, 2, 2] -
                  psf_sigma_bar[b, k, 1, 2] * psf_sigma_bar[b, k, 2, 1])
                  for b=1:CelesteTypes.B, s=1:mp.S, k=1:n_pcf_comp ];

    star_precision = zeros(Float64, CelesteTypes.B, mp.S, n_pcf_comp, 2, 2)
    for b=1:CelesteTypes.B, s=1:mp.S, k=1:n_pcf_comp
        star_precision[b, s, k, 1, 1] = psf_sigma_bar[b, k, 2, 2] / star_det[b, s, k]
        star_precision[b, s, k, 2, 2] = psf_sigma_bar[b, k, 1, 1] / star_det[b, s, k]
        star_precision[b, s, k, 1, 2] = -psf_sigma_bar[b, k, 1, 2] / star_det[b, s, k]
        star_precision[b, s, k, 2, 1] = -psf_sigma_bar[b, k, 2, 1] / star_det[b, s, k]
    end

    star_z = Float64[ psf_alpha_bar[b, k] ./ (star_det[b, s, k] ^ 0.5 * 2pi)
               for b=1:CelesteTypes.B, s=1:mp.S, k=1:n_pcf_comp ];

    # galaxy bvn components

    # Terms originally from Util.get_bvn_cov(rho, phi, sigma):

    #=
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
    =#

    #=
    @defNLExpr(galaxy_w_mat[s=1:mp.S, row=1:2, col=1:2],
               vp_sigma[s]*
               (sum{ cos(vp_phi[s]); row == 1 && col == 1} +
                sum{-vp_rho[s]*sin(vp_phi[s]); row == 2 && col == 1} +
                sum{ sin(vp_phi[s]); row == 1 && col == 2} +
                sum{ vp_rho[s]*cos(vp_phi[s]); row == 2 && col == 2}));
    =#
    # This is W' * W
    #=
    @defNLExpr(galaxy_xixi_mat[s=1:mp.S, xixi_row=1:2, xixi_col=1:2],
               sum{galaxy_w_mat[s, xixi_sum_index, xixi_row] *
                   galaxy_w_mat[s, xixi_sum_index, xixi_col],
                   xixi_sum_index = 1:2});
    =#

    # (2,1) : w[1,2]*w[1,1] + w[2,2]*w[2,1]
    # (1,2) : w[1,1]*w[1,2] + w[2,1]*w[2,2]
    # (2,2) : w[1,2]^2 + w[2,2]^2
    # This is W' * W
    @defNLExpr(galaxy_xixi_mat[s=1:mp.S, row=1:2, col=1:2],
               vp_sigma[s]^2*
               (sum{ 1 + (vp_rho[s]^2-1)*sin(vp_phi[s])^2 ; row == 1 && col == 1} +
                sum{ (1-vp_rho[s]^2)*sin(2*vp_phi[s])/2; row != col} +
                sum{ 1 + (vp_rho[s]^2-1)*cos(vp_phi[s])^2; row == 2 && col == 2}));

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
                    prec_row == 2 && prec_col == 1}))
                    # / galaxy_type1_det[b, s, k, g_k]);

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
                    prec_row == 2 && prec_col == 1}))
               # / galaxy_type2_det[b, s, k, g_k]);

    @defNLExpr(galaxy_type1_z[b=1:CelesteTypes.B, s=1:mp.S,
                              k=1:n_pcf_comp, g_k=1:n_gal1_comp],
                (galaxy_type1_alpha_tilde[g_k] * psf_alpha_bar[b, k]) ./
                (sqrt(galaxy_type1_det[b, s, k, g_k]) * 2pi));

    @defNLExpr(galaxy_type2_z[b=1:CelesteTypes.B, s=1:mp.S,
                              k=1:n_pcf_comp, g_k=1:n_gal2_comp],
               (galaxy_type2_alpha_tilde[g_k] * psf_alpha_bar[b, k]) ./
               (sqrt(galaxy_type2_det[b, s, k, g_k]) * 2pi));

    # Get the pdf values for each pixel.  Thie takes care of
    # the functions accum_galaxy_pos and accum_star_pos.

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

    # In this and similar expressions below, not every element will be evaluated -- later we will
    # sum it only over sources that are associated with a particular pixel.
    @defNLExpr(star_pdf_f[img=1:CelesteTypes.B, s=1:mp.S, k=1:n_pcf_comp,
                          pw=1:img_w, ph=1:img_h],
                    exp(-0.5 * sum{star_pdf_mean[img, s, k, pw, ph, pdf_f_row] * 
                                   star_precision[img, s, k, pdf_f_row, pdf_f_col] *
                                   star_pdf_mean[img, s, k, pw, ph, pdf_f_col],
                                   pdf_f_row=1:2, pdf_f_col=1:2}) *
                    star_z[img, s, k]
                 );

    # Galaxy pdfs
    @defNLExpr(galaxy_type1_pdf_f[img=1:CelesteTypes.B, s=1:mp.S,
                                  k=1:n_pcf_comp, g_k=1:n_gal1_comp,
                                  pw=1:img_w, ph=1:img_h],
            sum{
                exp(-0.5 * sum{star_pdf_mean[img, s, k, pw, ph, pdf_f_row] * 
                               galaxy_type1_precision[img, s, k, g_k, pdf_f_row, pdf_f_col] *
                               star_pdf_mean[img, s, k, pw, ph, pdf_f_col],
                               pdf_f_row=1:2, pdf_f_col=1:2}/
                               galaxy_type1_det[img, s, k, g_k]) *
                galaxy_type1_z[img, s, k, g_k]
             });

    @defNLExpr(galaxy_type2_pdf_f[img=1:CelesteTypes.B, s=1:mp.S,
                                  k=1:n_pcf_comp, g_k=1:n_gal2_comp,
                                  pw=1:img_w, ph=1:img_h],
                exp(-0.5 * sum{star_pdf_mean[img, s, k, pw, ph, pdf_f_row] * 
                               galaxy_type2_precision[img, s, k, g_k, pdf_f_row, pdf_f_col] *
                               star_pdf_mean[img, s, k, pw, ph, pdf_f_col],
                               pdf_f_row=1:2, pdf_f_col=1:2}/
                               galaxy_type2_det[img, s, k, g_k]
                               ) *
                galaxy_type2_z[img, s, k, g_k]
             );

    # Get the expectation and variance of G (accum_pixel_source_stats)

    # Star density values:
    @defNLExpr(fs0m[img=1:CelesteTypes.B, s=1:mp.S, pw=1:img_w, ph=1:img_h],
               sum{star_pdf_f[img, s, k, pw, ph], k=1:n_pcf_comp});

    # Galaxy density values:
    @defNLExpr(fs1m[img=1:CelesteTypes.B, s=1:mp.S, pw=1:img_w, ph=1:img_h],
               sum{vp_theta[s] * galaxy_type1_pdf_f[img, s, k, g_k, pw, ph],
                   k=1:n_pcf_comp, g_k=1:n_gal1_comp} +
               sum{(1 - vp_theta[s]) * galaxy_type2_pdf_f[img, s, k, g_k, pw, ph],
                   k=1:n_pcf_comp, g_k=1:n_gal2_comp});

    # TODO: how can you efficiently only evaluate E_G_s and Var_G only at select
    #       pixel X source values? 
    @defNLExpr(E_G_s[img=1:CelesteTypes.B, s=1:mp.S, pw=1:img_w, ph=1:img_h],
               (1 - vp_chi[s]) * E_l_a[s, img, 1] * fs0m[img, s, pw, ph] +
               vp_chi[s]       * E_l_a[s, img, 2] * fs1m[img, s, pw, ph]);

    @defNLExpr(Var_G_s[img=1:CelesteTypes.B, s=1:mp.S, pw=1:img_w, ph=1:img_h],
               (1 - vp_chi[s]) * E_ll_a[s, img, 1] * (fs0m[img, s, pw, ph] ^ 2) +
               vp_chi[s]       * E_ll_a[s, img, 2] * (fs1m[img, s, pw, ph] ^ 2) -
               (E_G_s[img, s, pw, ph] ^ 2));

    # Sum only over the sources associated with a particular pixel.
    @defNLExpr(E_G[img=1:CelesteTypes.B, pw=1:img_w, ph=1:img_h],
               sum{E_G_s[img, s, pw, ph], s=pixel_sources[pw, ph]} + blob_epsilon[img]);

    @defNLExpr(Var_G[img=1:CelesteTypes.B, pw=1:img_w, ph=1:img_h],
               sum{Var_G_s[img, s, pw, ph], s=pixel_sources[pw, ph]});

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

    jump_m, elbo_log_likelihood
end

end
