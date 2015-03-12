using JuMP

#module ElboJuMP

using Celeste
using CelesteTypes
import Util
import JuMP.getValue
import SampleData
using Base.Test

# Use this until getValue for nonlinear expressions is in JuMP proper.
# Right now this doesn't work for arrays.
function getValue(x::ReverseDiffSparse.ParametricExpression, m::Model)
	ReverseDiffSparse.getvalue(x, m.colVal)
end


# Set the JuMP parameters using the mp object
function SetJuMPParameters(mp::ModelParams)
	for s=1:mp.S
		setValue(vp_chi[s], mp.vp[s][ids.chi])
		setValue(vp_mu[s, 1], mp.vp[s][ids.mu][1])
		setValue(vp_mu[s, 2], mp.vp[s][ids.mu][2])

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


# For now, global constants accessed within the expressions
blobs, mp, three_bodies = SampleData.gen_three_body_dataset();

celeste_m = Model()


##########################
# Define the variational parameters.

# One set of variational parameters for each object.
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



########################
# Define the source brightness terms.

SetJuMPParameters(mp)

# Index 3 is r_s and  has a gamma expectation.
@defNLExpr(E_l_a_3[s=1:mp.S, a=1:CelesteTypes.I],
	       vp_gamma[s, a] * vp_zeta[s, a])

# The remaining indices involve c_s and have lognormal
# expectations times E_c_3.
@defNLExpr(E_l_a_4[s=1:mp.S, a=1:CelesteTypes.I],
           E_l_a_3[s, a] * exp(vp_beta[s, 3, a] + .5 * vp_lambda[s, 3, a]))
@defNLExpr(E_l_a_5[s=1:mp.S, a=1:CelesteTypes.I],
           E_l_a_4[s, a] * exp(vp_beta[s, 4, a] + .5 * vp_lambda[s, 4, a]))

@defNLExpr(E_l_a_2[s=1:mp.S, a=1:CelesteTypes.I],
           E_l_a_3[s, a] * exp(-vp_beta[s, 2, a] + .5 * vp_lambda[s, 2, a]))
@defNLExpr(E_l_a_1[s=1:mp.S, a=1:CelesteTypes.I],
           E_l_a_2[s, a] * exp(-vp_beta[s, 1, a] + .5 * vp_lambda[s, 1, a]))

# Copy the brightnesses into a summable indexed structure.
@defNLExpr(E_l_a[s=1:mp.S, b=1:CelesteTypes.B, a=1:CelesteTypes.I],
	       (b == 1) * E_l_a_1[s, a] +
	       (b == 2) * E_l_a_2[s, a] +
	       (b == 3) * E_l_a_3[s, a] +
	       (b == 4) * E_l_a_4[s, a] +
	       (b == 5) * E_l_a_5[s, a])

jump_e_l_a = [ ReverseDiffSparse.getvalue(E_l_a[s, b, a], celeste_m.colVal)
               for s=1:mp.S, a=1:CelesteTypes.I, b=1:CelesteTypes.B ]

celeste_e_l_a = [ ElboDeriv.SourceBrightness(mp.vp[s]).E_l_a[b, a].v
                  for s=1:mp.S, a=1:CelesteTypes.I, b=1:CelesteTypes.B ]


# Second order terms.
@defNLExpr(E_ll_a_3[s=1:mp.S, a=1:CelesteTypes.I],
	       vp_gamma[s, a] * (1 + vp_gamma[s, a]) * vp_zeta[s, a] ^ 2)

@defNLExpr(E_ll_a_4[s=1:mp.S, a=1:CelesteTypes.I],
	       E_ll_a_3[s, a] * exp(2 * vp_beta[s, 3, a] + 2 * vp_lambda[s, 3, a]))
@defNLExpr(E_ll_a_5[s=1:mp.S, a=1:CelesteTypes.I],
	       E_ll_a_4[s, a] * exp(2 * vp_beta[s, 4, a] + 2 * vp_lambda[s, 4, a]))

@defNLExpr(E_ll_a_2[s=1:mp.S, a=1:CelesteTypes.I],
	       E_ll_a_3[s, a] * exp(-2 * vp_beta[s, 2, a] + 2 * vp_lambda[s, 2, a]))
@defNLExpr(E_ll_a_1[s=1:mp.S, a=1:CelesteTypes.I],
	       E_ll_a_2[s, a] * exp(-2 * vp_beta[s, 1, a] + 2 * vp_lambda[s, 1, a]))

@defNLExpr(E_ll_a[s=1:mp.S, b=1:CelesteTypes.B, a=1:CelesteTypes.I],
	       (b == 1) * E_ll_a_1[s, a] +
	       (b == 2) * E_ll_a_2[s, a] +
	       (b == 3) * E_ll_a_3[s, a] +
	       (b == 4) * E_ll_a_4[s, a] +
	       (b == 5) * E_ll_a_5[s, a])


jump_e_ll_a = [ ReverseDiffSparse.getvalue(E_ll_a[s, b, a], celeste_m.colVal)
               for s=1:mp.S, a=1:CelesteTypes.I, b=1:CelesteTypes.B ]

celeste_e_ll_a = [ ElboDeriv.SourceBrightness(mp.vp[s]).E_ll_a[b, a].v
                  for s=1:mp.S, a=1:CelesteTypes.I, b=1:CelesteTypes.B ]

jump_e_ll_a - celeste_e_ll_a


####################################
# The bivariate normal mixtures, originally defined in load_bvn_mixtures

# The number of gaussian components in the gaussian mixture representations
# of the PCF.
const n_pcf_comp = 3

# Below I use the fact that the number of colors is also the number
# of images in a blob.

# These list comprehensions are necessary because JuMP can't index
# into immutable objects, it seems.
psf_xi_bar = [ blobs[b].psf[k].xiBar[row]
 		   for b=1:CelesteTypes.B, k=1:n_pcf_comp, row=1:2 ]
psf_sigma_bar = [ blobs[b].psf[k].SigmaBar[row, col]
                  for b=1:CelesteTypes.B, k=1:n_pcf_comp, row=1:2, col=1:2 ]
psf_alpha_bar = [ blobs[b].psf[k].alphaBar
                  for b=1:CelesteTypes.B, k=1:n_pcf_comp ]

@defNLExpr(star_mean[b=1:CelesteTypes.B, s=1:mp.S, k=1:n_pcf_comp, row=1:2],
	       psf_xi_bar[b, k, row] + vp_mu[s, row])

@defNLExpr(star_det[b=1:CelesteTypes.B, s=1:mp.S, k=1:n_pcf_comp],
	       psf_sigma_bar[b, k, 1, 1] * psf_sigma_bar[b, k, 2, 2] -
	       psf_sigma_bar[b, k, 1, 2] * psf_sigma_bar[b, k, 2, 1])

# Matrix inversion by hand.  Maybe it would be better to have a super-variable
# indexing all bivariate normals so as not to repeat code, or to do some
# macro magic to build the same nonlinear expression for each inverse. 
@defNLExpr(star_precision[b=1:CelesteTypes.B,
	                      s=1:mp.S, k=1:n_pcf_comp, row=1:2, col=1:2],
           (sum{psf_sigma_bar[b, k, 2, 2]; row == 1 && col == 1} +
           	sum{psf_sigma_bar[b, k, 1, 1]; row == 2 && col == 2} -
           	sum{psf_sigma_bar[b, k, 1, 2]; row == 1 && col == 2} -
           	sum{psf_sigma_bar[b, k, 2, 1]; row == 2 && col == 1}) / star_det[b, s, k])


@defNLExpr(star_z[b=1:CelesteTypes.B, s=1:mp.S, k=1:n_pcf_comp],
	       psf_alpha_bar[b, k] ./ (star_det[b, s, k] ^ 0.5 * 2pi))


#####################
# galaxy bvn components

# Terms from Util.get_bvn_cov(rho, phi, sigma):
@defNLExpr(star_precision[b=1:CelesteTypes.B,
	                      s=1:mp.S, k=1:n_pcf_comp, row=1:2, col=1:2],
           (sum{psf_sigma_bar[b, k, 2, 2]; row == 1 && col == 1} +
           	sum{psf_sigma_bar[b, k, 1, 1]; row == 2 && col == 2} -
           	sum{psf_sigma_bar[b, k, 1, 2]; row == 1 && col == 2} -
           	sum{psf_sigma_bar[b, k, 2, 1]; row == 2 && col == 1}) / star_det[b, s, k])

# This is R
@defNLExpr(galaxy_rot_mat[s=1:mp.S, row=1:2, col=1:2],
		   (sum{ cos(vp_rho[s]); row == 1 && col == 1} +
			sum{-sin(vp_rho[s]); row == 1 && col == 2} +
			sum{ sin(vp_rho[s]); row == 2 && col == 1} +
			sum{ cos(vp_rho[s]); row == 2 && col == 2}))

# This is D
@defNLExpr(galaxy_scale_mat[s=1:mp.S, row=1:2, col=1:2],
			sum{1.0; row == 1 && col == 1} +
			sum{0.0; row == 1 && col == 2} +
			sum{0.0; row == 2 && col == 1} +
			sum{vp_phi[s]; row == 2 && col == 2})

# This is scale * D * R'
@defNLExpr(galaxy_w_mat[s=1:mp.S, row=1:2, col=1:2],
		   vp_sigma[s] * sum{galaxy_scale_mat[s, row, sum_index] *
		                     galaxy_rot_mat[s, col, sum_index],
		                     sum_index = 1:2})

# This is W' * W
@defNLExpr(galaxy_xixi_mat[s=1:mp.S, row=1:2, col=1:2],
		   sum{galaxy_w_mat[s, sum_index, row] *
	           galaxy_w_mat[s, sum_index, col],
	           sum_index = 1:2})


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

# Terms from GalaxyCacheComponent:

# var_s and weight for type 1 galaxies:
@defNLExpr(galaxy_type1_var_s[b=1:CelesteTypes.B, s=1:mp.S,
	                          k=1:n_pcf_comp, g_k=1:n_gal1_comp,
	                          row=1:2, col=1:2],
	       psf_sigma_bar[b, k, row, col] +
	       galaxy_type1_sigma_tilde[g_k] * galaxy_xixi_mat[s, row, col])

@defNLExpr(galaxy_type1_weight[b=1:CelesteTypes.B,
	                           k=1:n_pcf_comp, g_k=1:n_gal1_comp],
	       psf_alpha_bar[b, k] * galaxy_type1_alpha_tilde[g_k])

# var_s and weight for type 2 galaxies:
@defNLExpr(galaxy_type2_var_s[b=1:CelesteTypes.B, s=1:mp.S,
	                          k=1:n_pcf_comp, g_k=1:n_gal2_comp,
	                          row=1:2, col=1:2],
	       psf_sigma_bar[b, k, row, col] +
	       galaxy_type2_sigma_tilde[g_k] * galaxy_xixi_mat[s, row, col])

@defNLExpr(galaxy_type2_weight[b=1:CelesteTypes.B,
	                           k=1:n_pcf_comp, g_k=1:n_gal2_comp],
	       psf_alpha_bar[b, k] * galaxy_type2_alpha_tilde[g_k])

# Now put these together to get the bivariate normal components,
# just like for the stars:

# The means are the same as for the stars.

# The determinant:
@defNLExpr(galaxy_type1_det[b=1:CelesteTypes.B, s=1:mp.S,
	                        k=1:n_pcf_comp, g_k=1:n_gal1_comp],
	       galaxy_type1_var_s[b, s, k, g_k, 1, 1] *
	       galaxy_type1_var_s[b, s, k, g_k, 2, 2] -
	       galaxy_type1_var_s[b, s, k, g_k, 1, 2] *
	       galaxy_type1_var_s[b, s, k, g_k, 2, 1])

@defNLExpr(galaxy_type2_det[b=1:CelesteTypes.B, s=1:mp.S,
	                        k=1:n_pcf_comp, g_k=1:n_gal2_comp],
	       galaxy_type2_var_s[b, s, k, g_k, 1, 1] *
	       galaxy_type2_var_s[b, s, k, g_k, 2, 2] -
	       galaxy_type2_var_s[b, s, k, g_k, 1, 2] *
	       galaxy_type2_var_s[b, s, k, g_k, 2, 1])

# Matrix inversion by hand.
@defNLExpr(galaxy_type1_precision[b=1:CelesteTypes.B, s=1:mp.S,
	                              k=1:n_pcf_comp, g_k=1:n_gal1_comp,
	                              row=1:2, col=1:2],
           (sum{galaxy_type1_var_s[b, s, k, g_k, 2, 2]; row == 1 && col == 1} +
           	sum{galaxy_type1_var_s[b, s, k, g_k, 1, 1]; row == 2 && col == 2} -
           	sum{galaxy_type1_var_s[b, s, k, g_k, 1, 2]; row == 1 && col == 2} -
           	sum{galaxy_type1_var_s[b, s, k, g_k, 2, 1]; row == 2 && col == 1}) /
           galaxy_type1_det[b, s, k, g_k])

@defNLExpr(galaxy_type2_precision[b=1:CelesteTypes.B, s=1:mp.S,
	                              k=1:n_pcf_comp, g_k=1:n_gal2_comp,
	                              row=1:2, col=1:2],
           (sum{galaxy_type2_var_s[b, s, k, g_k, 2, 2]; row == 1 && col == 1} +
           	sum{galaxy_type2_var_s[b, s, k, g_k, 1, 1]; row == 2 && col == 2} -
           	sum{galaxy_type2_var_s[b, s, k, g_k, 1, 2]; row == 1 && col == 2} -
           	sum{galaxy_type2_var_s[b, s, k, g_k, 2, 1]; row == 2 && col == 1}) /
           galaxy_type2_det[b, s, k, g_k])


@defNLExpr(galaxy_type1_z[b=1:CelesteTypes.B, s=1:mp.S,
	                      k=1:n_pcf_comp, g_k=1:n_gal1_comp],
	       psf_alpha_bar[b, k] ./ (galaxy_type1_det[b, s, k, g_k] ^ 0.5 * 2pi))

@defNLExpr(galaxy_type2_z[b=1:CelesteTypes.B, s=1:mp.S,
	                      k=1:n_pcf_comp, g_k=1:n_gal2_comp],
	       psf_alpha_bar[b, k] ./ (galaxy_type2_det[b, s, k, g_k] ^ 0.5 * 2pi))


###################
# Check everything:

star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(blobs[1].psf, mp)


celeste_star_mean = [
    ElboDeriv.load_bvn_mixtures(blobs[b].psf, mp)[1][k, s].the_mean[row]
    for b=1:CelesteTypes.B, s=1:mp.S, k=1:n_pcf_comp, row=1:2 ]
jump_star_mean =
	[ ReverseDiffSparse.getvalue(star_mean[b, s, k, row],
   							  celeste_m.colVal)
       for b=1:CelesteTypes.B, s=1:mp.S, k=1:n_pcf_comp, row=1:2 ]

celeste_star_precision = [
    ElboDeriv.load_bvn_mixtures(blobs[b].psf, mp)[1][k, s].precision[row, col]
    for b=1:CelesteTypes.B, s=1:mp.S, k=1:n_pcf_comp, row=1:2, col=1:2 ]
jump_star_precision =
	[ ReverseDiffSparse.getvalue(star_precision[b, s, k, row, col],
   							     celeste_m.colVal)
       for b=1:CelesteTypes.B, s=1:mp.S, k=1:n_pcf_comp, row=1:2, col=1:2 ]

celeste_star_precision - jump_star_precision

celeste_star_z = [
    ElboDeriv.load_bvn_mixtures(blobs[b].psf, mp)[1][k, s].z
    for b=1:CelesteTypes.B, s=1:mp.S, k=1:n_pcf_comp ]
jump_star_z =
	[ ReverseDiffSparse.getvalue(star_z[b, s, k],
   							     celeste_m.colVal)
       for b=1:CelesteTypes.B, s=1:mp.S, k=1:n_pcf_comp ]


