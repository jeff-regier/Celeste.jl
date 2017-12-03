"""
Calculate value, gradient, and hessian of the variational ELBO.
"""
module DeterministicVI

using Base.Threads: threadid, nthreads

import ..Config
using ..BivariateNormals: BivariateNormalDerivatives, BvnComponent,
                          GalaxySigmaDerivs, get_bvn_cov, eval_bvn_pdf!,
                          get_bvn_derivs!, transform_bvn_derivs!
using ..Model
using ..Model: ImagePatch, BvnBundle
import ..Celeste: Const, @aliasscope, @unroll_loop
using ..SensitiveFloats
import ..Log
using ..Transform
import DataFrames
import Optim
import ForwardDiff.Dual
using StaticArrays
import Base: convert

export ElboArgs, generic_init_source, catalog_init_source, init_sources,
       VariationalParams, elbo, ElboIntermediateVariables

function init_thread_pool!(pool::Vector, create)
    if length(pool) != Base.Threads.nthreads()
        empty!(pool)
        for i = 1:Base.Threads.nthreads()
            push!(pool, create())
        end
    end
end

"""
Return a default-initialized VariationalParams instance.
"""
function generic_init_source(init_pos::Vector{Float64})
    ret = Vector{Float64}(length(CanonicalParams))
    ret[ids.is_star] = 0.5
    ret[ids.pos] = init_pos
    ret[ids.flux_loc] = log(2.0)
    ret[ids.flux_scale] = 1e-3
    ret[ids.gal_frac_dev] = 0.5
    ret[ids.gal_axis_ratio] = 0.5
    ret[ids.gal_angle] = 0.0
    ret[ids.gal_radius_px] = 1.0
    ret[ids.k] = 1.0 / size(ids.k, 1)
    ret[ids.color_mean] = 0.0
    ret[ids.color_var] =  1e-2
    ret
end


"""
Return VariationalParams instance initialized form a catalog entry
"""
function catalog_init_source(ce::CatalogEntry; max_gal_radius_px=Inf)
    # TODO: sync this up with the transform bounds
    ret = generic_init_source(ce.pos)

    # TODO: don't do this thresholding for background sources,
    # just for sources that are being optimized
    ret[ids.is_star[1]] = ce.is_star ? 0.8: 0.2
    ret[ids.is_star[2]] = ce.is_star ? 0.2: 0.8

    ret[ids.flux_loc[1]] = log(max(0.1, ce.star_fluxes[3]))
    ret[ids.flux_loc[2]] = log(max(0.1, ce.gal_fluxes[3]))

    function get_color(color_var, color_mean)
        color_var > 0 && color_mean > 0 ? min(max(log(color_var / color_mean), -9.), 9.) :
            color_var > 0 && color_mean <= 0 ? 3.0 :
                color_var <= 0 && color_mean > 0 ? -3.0 : 0.0
    end

    function get_colors(raw_fluxes)
        [get_color(raw_fluxes[c+1], raw_fluxes[c]) for c in 1:4]
    end

    ret[ids.color_mean[:, 1]] = get_colors(ce.star_fluxes)
    ret[ids.color_mean[:, 2]] = get_colors(ce.gal_fluxes)

    ret[ids.gal_frac_dev] = min(max(ce.gal_frac_dev, 0.015), 0.985)

    ret[ids.gal_axis_ratio] = ce.is_star ? .8 : min(max(ce.gal_axis_ratio, 0.015), 0.985)
    ret[ids.gal_angle] = ce.gal_angle
    ret[ids.gal_radius_px] = ce.is_star ? 0.2 : min(max_gal_radius_px, max(ce.gal_radius_px, 0.2))

    ret
end


function init_sources(target_sources::Vector{Int}, catalog::Vector{CatalogEntry})
    ret = Vector{Vector{Float64}}(length(catalog))
    for s in 1:length(catalog)
        ret[s] = catalog_init_source(catalog[s])
    end
    for s in target_sources
        ret[s][:] = generic_init_source(catalog[s].pos)
    end
    ret
end


include("deterministic_vi/elbo_args.jl")
include("deterministic_vi/elbo_kl.jl")
include("deterministic_vi/source_brightness.jl")
include("deterministic_vi/elbo_objective.jl")
include("deterministic_vi/ConstraintTransforms.jl")
include("deterministic_vi/ElboMaximize.jl")

end
