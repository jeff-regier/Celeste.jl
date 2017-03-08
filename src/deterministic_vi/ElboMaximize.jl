module ElboMaximize

include(joinpath(Pkg.dir("Celeste"), "src", "cg_trust_region.jl"))

using Optim
using Optim: Options
using ..Model
using ..SensitiveFloats
using ..DeterministicVI: ElboArgs, elbo, ElboIntermediateVariables, VariationalParams
using ..DeterministicVI.ConstraintTransforms: TransformJacobianBundle,
                                              VariationalParams,
                                              ConstraintBatch,
                                              ParameterConstraint,
                                              BoxConstraint,
                                              SimplexConstraint,
                                              u_ParameterConstraints,
                                              allocate_free_params,
                                              to_free!,
                                              to_bound!,
                                              enforce!,
                                              propagate_derivatives!

import ForwardDiff
import ForwardDiff: Dual

##################################################
# defaults for optional arguments to `maximize!` #
##################################################

immutable Config{N,T}
    # working memory
    bound_params::VariationalParams{T}
    free_params::VariationalParams{T}
    jacobian_bundle::TransformJacobianBundle{N,T}
    objective_elbo_vars::ElboIntermediateVariables{T}
    gradient_elbo_vars::ElboIntermediateVariables{T}
    sf_free::SensitiveFloat{T}

    dual_vp::VariationalParams{Dual{1,T}}
    dual_bound_params::VariationalParams{Dual{1,T}}
    dual_free_params::VariationalParams{Dual{1,T}}
    dual_jacobian_bundle::TransformJacobianBundle{N,Dual{1,T}}
    hessvec_elbo_vars::ElboIntermediateVariables{Dual{1,T}}
    dual_sf_free::SensitiveFloat{Dual{1,T}}

    # settings
    constraints::ConstraintBatch
    optim_options::Options{Void}
    trust_region::CGTrustRegion{T}
end

function Config{T}(ea::ElboArgs,
                   vp::VariationalParams{T},
                   bound_params::VariationalParams{T} = vp[ea.active_sources];
                   loc_width::Float64 = 1e-4,
                   loc_scale::Float64 = 1.0,
                   constraints::ConstraintBatch = ConstraintBatch(bound_params, loc_width, loc_scale),
                   max_iters::Int = 50,
                   optim_options::Options = custom_optim_options(max_iters = max_iters),
                   trust_region::CGTrustRegion = custom_trust_region())
    free_params = allocate_free_params(bound_params, constraints)
    n_active_sources = length(bound_params)

    n_free_params = length(free_params[1])
    n_bound_params = length(bound_params[1])
    jacobian_bundle = TransformJacobianBundle(bound_params, free_params)
    objective_elbo_vars = ElboIntermediateVariables(T, ea.S, n_active_sources,
                                                    false, false)
    gradient_elbo_vars = ElboIntermediateVariables(T, ea.S, n_active_sources,
                                                   true, false)
    sf_free = SensitiveFloat{T}(n_free_params, n_bound_params, true, false)

    dual_vp = Vector{Dual{1,T}}[similar(src, Dual{1,T}) for src in vp]
    dual_bound_params = dual_vp[ea.active_sources]
    dual_free_params = Vector{Dual{1,T}}[similar(x, Dual{1,T}) for x in free_params]
    dual_jacobian_bundle = TransformJacobianBundle(dual_bound_params, dual_free_params)
    hessvec_elbo_vars = ElboIntermediateVariables(Dual{1,T}, ea.S, n_active_sources,
                                                  true, false)
    dual_sf_free = SensitiveFloat{Dual{1,T}}(n_free_params, n_bound_params, true, false)

    return Config(bound_params, free_params, jacobian_bundle,
                  objective_elbo_vars, gradient_elbo_vars, sf_free,
                  dual_vp, dual_bound_params, dual_free_params,
                  dual_jacobian_bundle, hessvec_elbo_vars, dual_sf_free,
                  constraints, optim_options, trust_region)
end

function custom_optim_options(; xtol_abs = 1e-7, ftol_rel = 1e-6, max_iters = 50)
    return Optim.Options(x_tol = xtol_abs, f_tol = ftol_rel, g_tol = 1e-8,
                         iterations = max_iters, store_trace = false,
                         show_trace = false, extended_trace = false)
end

function custom_trust_region(; initial_radius = 10.0, max_radius = 1e9)
    return CGTrustRegion(initial_radius = initial_radius,
                         max_radius = max_radius)
end

function enforce_references!{T}(ea::ElboArgs, vp::VariationalParams{T}, cfg::Config)
    @assert length(cfg.dual_vp) == length(vp)
    @assert length(cfg.bound_params) == length(ea.active_sources)
    zero_partial = ForwardDiff.Partials{1,T}(tuple(zero(T)))
    for src in 1:length(vp)
        vp_src, dual_vp_src = vp[src], cfg.dual_vp[src]
        for i in 1:length(vp_src)
            dual_vp_src[i] = Dual{1,T}(vp_src[i], zero_partial)
        end
    end
    for i in 1:length(ea.active_sources)
        cfg.bound_params[i] = vp[ea.active_sources[i]]
        cfg.dual_bound_params[i] = cfg.dual_vp[ea.active_sources[i]]
    end
    return nothing
end

##########################
# to_vector/from_vector! #
##########################

to_vector{T}(sources::VariationalParams{T}) = vcat(sources...)::Vector{T}

function from_vector!(sources, x)
    i = 1
    for src in sources
        for j in eachindex(src)
            src[j] = x[i]
            i += 1
        end
    end
    return sources
end

function dual_from_vector!(sources, x, v)
    i = 1
    for src in sources
        for j in 1:length(src)
            src[j] = Dual(x[i], v[i])
            i += 1
        end
    end
    return sources
end

##################################
# Callable Types Passed to Optim #
##################################

# Objective #
#-----------#

immutable Objective{F,N,T} <: Function
    f::F
    ea::ElboArgs
    vp::VariationalParams{T}
    cfg::Config{N,T}
end

"""
An unconstrained version of the ELBO
"""
function (f::Objective{F}){F}(x::Vector)
    from_vector!(f.cfg.free_params, x)
    to_bound!(f.cfg.bound_params, f.cfg.free_params, f.cfg.constraints)
    sf_bound = f.f(f.ea, f.vp, f.cfg.objective_elbo_vars)
    return -(sf_bound.v[])
end

# Gradient #
#----------#

immutable Gradient{F,N,T} <: Function
    f::F
    ea::ElboArgs
    vp::VariationalParams{T}
    cfg::Config{N,T}
end

"""
Computes both the gradient the objective function's value
"""
function (f::Gradient{F}){F}(x::Vector, result::Vector)
    from_vector!(f.cfg.free_params, x)
    to_bound!(f.cfg.bound_params, f.cfg.free_params, f.cfg.constraints)
    sf_bound = f.f(f.ea, f.vp, f.cfg.gradient_elbo_vars)
    propagate_derivatives!(to_bound!, sf_bound, f.cfg.sf_free, f.cfg.free_params,
                           f.cfg.constraints, f.cfg.jacobian_bundle)
    free_gradient = f.cfg.sf_free.d
    for i in eachindex(result)
        result[i] = -(free_gradient[i])
    end
    return -(f.cfg.sf_free.v[])
end

# HessianVectorProduct #
#----------------------#

immutable HessianVectorProduct{F,N,T} <: Function
    f::F
    ea::ElboArgs
    vp::VariationalParams{T}
    cfg::Config{N,T}
end

"""
Computes hessian-vector products
"""
function (f::HessianVectorProduct{F}){F}(x::Vector, v::Vector, result::Vector)
    dual_from_vector!(f.cfg.dual_free_params, x, v)
    to_bound!(f.cfg.dual_bound_params, f.cfg.dual_free_params, f.cfg.constraints)
    dual_sf_bound = f.f(f.ea, f.cfg.dual_vp, f.cfg.hessvec_elbo_vars)
    propagate_derivatives!(to_bound!, dual_sf_bound, f.cfg.dual_sf_free, f.cfg.dual_free_params,
                           f.cfg.constraints, f.cfg.dual_jacobian_bundle)
    free_gradient = f.cfg.dual_sf_free.d
    for i in eachindex(result)
        result[i] = -(ForwardDiff.partials(free_gradient[i], 1))
    end
    return nothing
end

#############
# maximize! #
#############

function maximize!{F}(f::F, ea::ElboArgs, vp::VariationalParams{Float64}, cfg::Config = Config(ea, vp))
    enforce_references!(ea, vp, cfg)
    enforce!(cfg.bound_params, cfg.constraints)
    to_free!(cfg.free_params, cfg.bound_params, cfg.constraints)
    x = to_vector(cfg.free_params)

    ddf = TwiceDifferentiableHV(Objective(f, ea, vp, cfg),
                                Gradient(f, ea, vp, cfg),
                                HessianVectorProduct(f, ea, vp, cfg))
    R = Optim.MultivariateOptimizationResults{Float64,1,CGTrustRegion{Float64}}
    result::R = Optim.optimize(ddf, x, cfg.trust_region, cfg.optim_options)

    min_value::Float64 = -(Optim.minimum(result))
    min_solution::Vector{Float64} = Optim.minimizer(result)
    from_vector!(cfg.free_params, min_solution)
    to_bound!(cfg.bound_params, cfg.free_params, cfg.constraints)

    return result.f_calls::Int64, min_value, min_solution, result
end

@inline function maximize!(ea::ElboArgs, vp::VariationalParams{Float64}, cfg::Config = Config(ea, vp))
    return maximize!(elbo, ea, vp, cfg)
end

end # module
