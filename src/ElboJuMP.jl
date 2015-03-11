using JuMP

#module ElboJuMP

using Celeste
using CelesteTypes
import Util
import JuMP.getValue
import SampleData

# Use this until getValue for nonlinear expressions is in JuMP proper.
function getValue(x::ReverseDiffSparse.ParametricExpression, m::Model)
	ReverseDiffSparse.getvalue(x, m.colVal)
end

# For now, global constants accessed within the expressions
blobs, mp, three_bodies = SampleData.gen_three_body_dataset()

celeste_m = Model()

# One set of variational parameters for each object.
# These replace the ModelParams.vp object in the old version.

# The probability of being a galaxy.  (0 = star, 1 = galaxy)
@defVar(celeste_m, 0  <= chi[s=1:mp.S] <= 1)

 # The location of the object (2x1 vector)
@defVar(celeste_m, mu[s=1:mp.S, a=1:CelesteTypes.I])

# Ix1 scalar variational parameters for r_s.  The first
# row is for stars, and the second for galaxies (I think?).
@defVar(celeste_m, gamma[s=1:mp.S, i=1:CelesteTypes.I])
@defVar(celeste_m, zeta[s=1:mp.S,  i=1:CelesteTypes.I])

# The weight given to a galaxy of type 1.
@defVar(celeste_m, theta[s=1:mp.S])

# galaxy minor/major ratio
@defVar(celeste_m, rho[s=1:mp.S])

# galaxy angle
@defVar(celeste_m, phi[s=1:mp.S])

# galaxy scale
@defVar(celeste_m, sigma[s=1:mp.S])

# The remaining parameters are matrices where the
# first column is for stars and the second is for galaxies.

# DxI matrix of color prior component indicators.
@defVar(celeste_m, kappa[s=1:mp.S, d=1:CelesteTypes.D, a=1:CelesteTypes.I])

# (B - 1)xI matrices containing c_s means and variances, respectively.
@defVar(celeste_m, beta[s=1:mp.S, b=1:(CelesteTypes.B - 1), a=1:CelesteTypes.I])
@defVar(celeste_m, lambda[s=1:mp.S, b=1:(CelesteTypes.B - 1), a=1:CelesteTypes.I])
