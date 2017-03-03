module NewtonMaximize

include(joinpath(Pkg.dir("Celeste"), "src", "cg_trust_region.jl"))

using Optim
using Optim: Options
using ..Model
using ..SensitiveFloats
using ..DeterministicVI: ElboArgs, elbo, ElboIntermediateVariables, VariationalParams
using ...ConstraintTransforms: TransformDerivatives,
                               ParameterBatch,
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
import ForwardDiff: Dual


##################################################
# defaults for optional arguments to `maximize!` #
##################################################

immutable Config{N,T}
    bound_params::ParameterBatch{T}
    free_params::ParameterBatch{T}
    constraints::ConstraintBatch
    derivs::TransformDerivatives{N,T}
    optim_options::Options{Void}
    trust_region::CGTrustRegion{T}
end

function Config{T}(bound_params::ParameterBatch{T};
                   loc_width::Float64 = 1e-4,
                   loc_scale::Float64 = 1.0,
                   max_iters::Int = 50,
                   constraints::ConstraintBatch = ConstraintBatch(bound_params, loc_width, loc_scale),
                   free_params::ParameterBatch{T} = allocate_free_params(bound_params, constraints),
                   derivs::TransformDerivatives = TransformDerivatives(bound_params, free_params),
                   optim_options::Options = custom_optim_options(max_iters=max_iters),
                   trust_region::CGTrustRegion = custom_trust_region())
    return Config(bound_params, free_params, constraints, derivs,
                  optim_options, trust_region)
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

#############
# Objective #
#############


immutable Objective{N} <: Function
    ea::ElboArgs
    elbo_vars::ElboIntermediateVariables{Float64}
    sf_free::SensitiveFloat{Float64}
    cfg::Config{N,Float64}
end

immutable Gradient{N} <: Function
    ea::ElboArgs
    elbo_vars::ElboIntermediateVariables{Float64}
    sf_free::SensitiveFloat{Float64}
    cfg::Config{N,Float64}
end

immutable Hessian{N} <: Function
    ea::ElboArgs
    elbo_vars::ElboIntermediateVariables{Dual{1, Float64}}
    sf_free::SensitiveFloat{Dual{1, Float64}}
    cfg::Config{N,Dual{1, Float64}}
end

"""
An unconstrained version of the ELBO
"""
function Objective(ea::ElboArgs, cfg::Config, x::Vector)
    elbo_vars = ElboIntermediateVariables(Float64, ea.S, length(ea.active_sources); 
                                          calculate_gradient=calculate_gradient, 
                                          calculate_hessian=false)
    sf_free = SensitiveFloat{Float64}(length(cfg.free_params[1]),
                                      length(cfg.bound_params), false, false)
    return Objective(ea, elbo_vars, sf_free, cfg)
end

"""
Computes both the gradient the objective function's value
"""
function Gradient(ea::ElboArgs, cfg::Config, x::Vector)
    elbo_vars = ElboIntermediateVariables(Float64, ea.S, length(ea.active_sources); 
                                          calculate_gradient=calculate_gradient, 
                                          calculate_hessian=false)
    sf_free = SensitiveFloat{Float64}(length(cfg.free_params[1]),
                                      length(cfg.bound_params), true, false)
    return Gradient(ea, elbo_vars, sf_free, cfg)
end

"""
Computes hessian-vector products
"""
function Hessian(ea::ElboArgs, cfg::Config, x::Vector)
    elbo_vars = ElboIntermediateVariables(Dual{1, Float64}, ea.S, length(ea.active_sources); 
                                          calculate_gradient=true, 
                                          calculate_hessian=false)
    sf_free = SensitiveFloat{Dual{1, Float64}}(length(cfg.free_params[1]), 
                                    length(cfg.bound_params), true, false)
    return Hessian(ea, elbo_vars, sf_free, cfg)
end

function evaluate!(f::Objective, x::Vector)
    from_vector!(f.cfg.free_params, x)
    to_bound!(f.cfg.bound_params, f.cfg.free_params, f.cfg.constraints)
    sf_bound = f.f(f.ea)
    propagate_derivatives!(to_bound!, sf_bound, f.sf_free, f.cfg.free_params,
                           f.cfg.constraints, f.cfg.derivs)
    return nothing
end

function (f::Objective)(x::Vector)
    evaluate!(f, x)
    return -(f.sf_free.v[])
end

function (f::Gradient)(x::Vector, result::Vector)
    evaluate!(f, x)
    free_gradient = f.sf_free.d
    for i in eachindex(result)
        result[i] = -(free_gradient[i])
    end
    return -(f.sf_free.v[])
end

function (f::Hessian)(x::Vector, v::Vector, result::Vector)
    evaluate!(f, x)
    H = Matrix{Float64}(length(x), length(x))
    free_hessian = f.sf_free.h
    for i in eachindex(H)
        H[i] = -(free_hessian[i])
    end
    result[:] = H * v
    return result
end

#############
# maximize! #
#############

function maximize!(ea::ElboArgs, vp::VariationalParams{Float64}; cfg_kwargs...)
    vp_dual = convert(VariationalParams{Dual{1, Float64}}, vp)

    cfg = Config(vp[ea.active_sources]; cfg_kwargs...)
    enforce!(cfg.bound_params, cfg.constraints)
    to_free!(cfg.free_params, cfg.bound_params, cfg.constraints)
    x = to_vector(cfg.free_params)

    ddf = TwiceDifferentiableHV(Objective(ea, cfg, x),
                                Gradient(ea, cfg, x),
                                Hessian(ea, cfg, x))
    R = Optim.MultivariateOptimizationResults{T,1,CGTrustRegion{T}}
    result::R = Optim.optimize(ddf, x, cfg.trust_region, cfg.optim_options)

    min_value::T = -(Optim.minimum(result))
    min_solution::Vector{T} = Optim.minimizer(result)
    from_vector!(cfg.free_params, min_solution)
    to_bound!(cfg.bound_params, cfg.free_params, cfg.constraints)

    return result.f_calls::Int64, min_value, min_solution, result
end

to_vector{T}(sources::ParameterBatch{T}) = vcat(sources...)::Vector{T}

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

end # module
