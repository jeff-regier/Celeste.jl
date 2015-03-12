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
# of the PCF and galaxies.
const n_pcf_comp = 3
const n_gal_comp = 8

# The number of images in a blob.
const n_img = 5

# Not working:
@defNLExpr(star_mean[asdf=1:5, s=1:mp.S, k=1:n_pcf_comp, row=1:2],
	       blobs[1].psf[1].xiBar[row] + asdf + k)

@defNLExpr(star_mean[asdf=1:5], blobs[asdf].psf[1].xiBar[1])

@defNLExpr(star_mean[asdf=1:5, s=1:mp.S, k=1:n_pcf_comp, row=1:2],
	       blobs[asdf].psf[k].xiBar[row] + vp_mu[s, row])


@defNLExpr(star_det[s=1:mp.S, k=1:n_pcf_comp],
	       blobs[img].psf[k].SigmaBar[1, 1] * blobs[img].psf[k].SigmaBar[2, 2] -
	       blobs[img].psf[k].SigmaBar[1, 2] * blobs[img].psf[k].SigmaBar[2, 1])

# Matrix inversion by hand.  Maybe it would be better to have a super-variable
# indexing all bivariate normals so as not to repeat code, or to do some
# macro magic to build the same nonlinear expression for each inverse. 
@defNLExpr(star_precision[s=1:mp.S, k=1:n_pcf_comp, row=1:2, col=1:2],
           (sum{blobs[img].psf[k].SigmaBar[2, 2], row == 1, col == 1} +
           	sum{blobs[img].psf[k].SigmaBar[1, 1], row == 2, col == 2} -
           	sum{blobs[img].psf[k].SigmaBar[1, 2], row == 1, col == 2} -
           	sum{blobs[img].psf[k].SigmaBar[2, 1], row == 2, col == 1}) / star_det[s, k])

@defNLExpr(star_z[s=1:mp.S, k=1:n_pcf_comp],
	       blobs[img].psf[k].alphaBar ./ (star_det[s, k] ^ 0.5 * 2pi))
