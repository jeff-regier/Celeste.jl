using JuMP

module ElboJuMP

using CelesteJuMPTypes
import Util
import JuMP.getValue

# Use this until getValue for nonlinear expressions is in JuMP proper.
function getValue(x::ReverseDiffSparse.ParametricExpression, m::Model)
	ReverseDiffSparse.getvalue(x, m.colVal)
end

# Global constants accessed within the expressions
mp = ModelParams()

celeste_model = Model()

# One set of variational parameters for each object.
# These replace the ModelParams.vp object in the old version.
@defVar(celeste_model, )