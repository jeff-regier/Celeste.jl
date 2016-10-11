"""
Calculate value, gradient, and hessian of the variational ELBO.
"""
module DeterministicVI

using ..Model
import ..Model: BivariateNormalDerivatives, BvnComponent, GalaxyCacheComponent,
                GalaxySigmaDerivs,
                get_bvn_cov, eval_bvn_pdf!, get_bvn_derivs!,
                transform_bvn_derivs!
using ..SensitiveFloats
import ..SensitiveFloats.clear!
import ..Log
using ..Transform
import DataFrames
import Optim

export ElboArgs


include("deterministic_vi/elbo_args.jl")
include("deterministic_vi/elbo_kl.jl")
include("deterministic_vi/source_brightness.jl")
include("deterministic_vi/elbo_objective.jl")
include("deterministic_vi/maximize_elbo.jl")


end
