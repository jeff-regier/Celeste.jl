"""
Calculate value, gradient, and hessian of the variational ELBO.
"""
module DeterministicVI

using ..Model
import ..Model: BivariateNormalDerivatives, BvnComponent, GalaxyCacheComponent,
                GalaxySigmaDerivs, SkyPatch,
                get_bvn_cov, eval_bvn_pdf!, get_bvn_derivs!,
                transform_bvn_derivs!
import ..Infer
using ..SensitiveFloats
import ..SensitiveFloats.clear!
import ..Log
using ..Transform
import DataFrames
import Optim
using StaticArrays

export ElboArgs


include("deterministic_vi/elbo_args.jl")
include("deterministic_vi/elbo_kl.jl")
include("deterministic_vi/source_brightness.jl")
include("deterministic_vi/elbo_objective.jl")
include("deterministic_vi/maximize_elbo.jl")


"""
Infers one light source. This routine is intended to be called in parallel,
once per target light source.

Arguments:
    images: a collection of astronomical images
    neighbors: the other light sources near `entry`
    entry: the source to infer
"""
function infer_source(images::Vector{Image},
                      neighbors::Vector{CatalogEntry},
                      entry::CatalogEntry)
    if length(neighbors) > 100
        Log.warn("Excessive number ($(length(neighbors))) of neighbors")
    end

    # It's a bit inefficient to call the next 5 lines every time we optimize_f.
    # But, as long as runtime is dominated by the call to maximize_f, that
    # isn't a big deal.
    cat_local = vcat([entry], neighbors)
    vp = Vector{Float64}[init_source(ce) for ce in cat_local]
    patches = Infer.get_sky_patches(images, cat_local)
    Infer.load_active_pixels!(images, patches)

    ea = ElboArgs(images, vp, patches, [1])
    f_evals, max_f, max_x, nm_result = maximize_f(elbo, ea)
    vp[1]
end


end
