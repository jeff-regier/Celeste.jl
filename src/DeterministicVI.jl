"""
Calculate value, gradient, and hessian of the variational ELBO.
"""
module DeterministicVI

using Base.Threads: threadid, nthreads

import ..Configs
using ..Model
import ..Model: BivariateNormalDerivatives, BvnComponent, GalaxyCacheComponent,
                GalaxySigmaDerivs, SkyPatch,
                get_bvn_cov, eval_bvn_pdf!, get_bvn_derivs!,
                transform_bvn_derivs!, populate_fsm!
import ..Infer
using ..SensitiveFloats
import ..SensitiveFloats.clear!
import ..Log
using ..Transform
import DataFrames
import Optim
import ForwardDiff.Dual
using StaticArrays
import Base.convert

export ElboArgs, generic_init_source, catalog_init_source, init_sources


"""
Return a default-initialized VariationalParams instance.
"""
function generic_init_source(init_pos::Vector{Float64})
    ret = Vector{Float64}(length(CanonicalParams))
    ret[ids.a] = 0.5
    ret[ids.u] = init_pos
    ret[ids.r1] = log(2.0)
    ret[ids.r2] = 1e-3
    ret[ids.e_dev] = 0.5
    ret[ids.e_axis] = 0.5
    ret[ids.e_angle] = 0.0
    ret[ids.e_scale] = 1.0
    ret[ids.k] = 1.0 / size(ids.k, 1)
    ret[ids.c1] = 0.0
    ret[ids.c2] =  1e-2
    ret
end


"""
Return VariationalParams instance initialized form a catalog entry
"""
function catalog_init_source(ce::CatalogEntry; max_gal_scale=Inf)
    # TODO: sync this up with the transform bounds
    ret = generic_init_source(ce.pos)

    # TODO: don't do this thresholding for background sources,
    # just for sources that are being optimized
    ret[ids.a[1]] = ce.is_star ? 0.8: 0.2
    ret[ids.a[2]] = ce.is_star ? 0.2: 0.8

    ret[ids.r1[1]] = log(max(0.1, ce.star_fluxes[3]))
    ret[ids.r1[2]] = log(max(0.1, ce.gal_fluxes[3]))

    function get_color(c2, c1)
        c2 > 0 && c1 > 0 ? min(max(log(c2 / c1), -9.), 9.) :
            c2 > 0 && c1 <= 0 ? 3.0 :
                c2 <= 0 && c1 > 0 ? -3.0 : 0.0
    end

    function get_colors(raw_fluxes)
        [get_color(raw_fluxes[c+1], raw_fluxes[c]) for c in 1:4]
    end

    ret[ids.c1[:, 1]] = get_colors(ce.star_fluxes)
    ret[ids.c1[:, 2]] = get_colors(ce.gal_fluxes)

    ret[ids.e_dev] = min(max(ce.gal_frac_dev, 0.015), 0.985)

    ret[ids.e_axis] = ce.is_star ? .8 : min(max(ce.gal_ab, 0.015), 0.985)
    ret[ids.e_angle] = ce.gal_angle
    ret[ids.e_scale] = ce.is_star ? 0.2 : min(max_gal_scale, max(ce.gal_scale, 0.2))

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
include("deterministic_vi/NewtonMaximize.jl")


"""
Infers one light source. This routine is intended to be called in parallel,
once per target light source.

Arguments:
    images: a collection of astronomical images
    neighbors: the other light sources near `entry`
    entry: the source to infer
"""
function infer_source(config::Configs.Config,
                      images::Vector{Image},
                      neighbors::Vector{CatalogEntry},
                      entry::CatalogEntry)
    if length(neighbors) > 100
        msg = string("objid $(entry.objid) [ra: $(entry.pos)] has an excessive",
                     "number ($(length(neighbors))) of neighbors")
        Log.warn(msg)
    end

    # It's a bit inefficient to call the next 5 lines every time we optimize_f.
    # But, as long as runtime is dominated by the call to maximize!, that
    # isn't a big deal.
    cat_local = vcat([entry], neighbors)
    vp = init_sources([1], cat_local)
    patches = Infer.get_sky_patches(images, cat_local)
    Infer.load_active_pixels!(config, images, patches)

    ea = ElboArgs(images, patches, [1])
    f_evals, max_f, max_x, nm_result = NewtonMaximize.maximize!(elbo, ea)
    return vp[1]
end

# legacy wrapper
function infer_source(images::Vector{Image},
                      neighbors::Vector{CatalogEntry},
                      entry::CatalogEntry)
    infer_source(Configs.Config(), images, neighbors, entry)
end

end
