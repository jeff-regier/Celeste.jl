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

 # The location of the object (2x1 vector)
@defVar(celeste_m, vp_mu[s=1:mp.S, a=1:CelesteTypes.I])

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

jump_e_l_a = [ ReverseDiffSparse.getvalue(E_l_a[s, b, a], celeste_m.colVal)
               for s=1:mp.S, a=1:CelesteTypes.I, b=1:CelesteTypes.B ]

sb1 = ElboDeriv.SourceBrightness(mp.vp[s]);
celeste_val = sb1.E_l_a[b, a].v

# Index 3 is r_s and  has a gamma expectation.
@defNLExpr(E_l_a[s=1:mp.S, b=3, a=1:CelesteTypes.I],
	       vp_gamma[s, a] * vp_zeta[s, a])

# The remaining indices involve c_s and have lognormal
# expectations times E_c_3.
@defNLExpr(E_l_a[s=1:mp.S, b=4, a=1:CelesteTypes.I],
           E_l_a[s, 3, a] * exp(vp_beta[s, 3, a] + .5 * vp_lambda[s, 3, a]))

jump_val = ReverseDiffSparse.getvalue(E_l_a[s, b, a], celeste_m.colVal)
celeste_val == jump_val

@defNLExpr(E_l_a[s=1:mp.S, b=5, a=1:CelesteTypes.I],
           E_l_a[s, 4, a] * exp(vp_beta[s, 4, a] + .5 * vp_lambda[s, 4, a]))

@defNLExpr(E_l_a[s=1:mp.S, b=2, a=1:CelesteTypes.I],
           E_l_a[s, 3, a] * exp(-vp_beta[s, 2, a] + .5 * vp_lambda[s, 2, a]))
@defNLExpr(E_l_a[s=1:mp.S, b=1, a=1:CelesteTypes.I],
           E_l_a[s, 2, a] * exp(-vp_beta[s, 1, a] + .5 * vp_lambda[s, 1, a]))

# Second order terms.
@defNLExpr(E_ll_a[s=1:mp.S, b=3, a=1:CelesteTypes.I],
	       vp_gamma[s, a] * (1 + vp_gamma[s, a]) * vp_zeta[s, a] ^ 2)

@defNLExpr(E_ll_a[s=1:mp.S, b=4, a=1:CelesteTypes.I],
	       E_ll_a[s, 3, a] * exp(2 * vp_beta[s, 3, a] + 2 * vp_lambda[s, 3, a]))
@defNLExpr(E_ll_a[s=1:mp.S, b=5, a=1:CelesteTypes.I],
	       E_ll_a[s, 4, a] * exp(2 * vp_beta[s, 4, a] + 2 * vp_lambda[s, 4, a]))

@defNLExpr(E_ll_a[s=1:mp.S, b=2, a=1:CelesteTypes.I],
	       E_ll_a[s, 3, a] * exp(-2 * vp_beta[s, 2, a] + 2 * vp_lambda[s, 2, a]))
@defNLExpr(E_ll_a[s=1:mp.S, b=1, a=1:CelesteTypes.I],
	       E_ll_a[s, 2, a] * exp(-2 * vp_beta[s, 1, a] + 2 * vp_lambda[s, 1, a]))


for s=1:mp.S
	for b=1:CelesteTypes.B
		for a=1:CelesteTypes.I
			print(s, ", ", b, ", ", a, "\n")

		end
	end
end


