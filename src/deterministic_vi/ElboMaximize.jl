module ElboMaximize

using Optim
using Optim: Options, NewtonTrustRegion
using ..Model
using ..SensitiveFloats
using ..DeterministicVI
using ..DeterministicVI.ConstraintTransforms: TransformDerivatives,
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

##################################################
# defaults for optional arguments to `maximize!` #
##################################################

immutable Config{N,T}
    vp::VariationalParams{T}
    bound_params::VariationalParams{T}
    free_params::VariationalParams{T}
    constraints::ConstraintBatch
    derivs::TransformDerivatives{N,T}
    optim_options::Options{Void}
    trust_region::NewtonTrustRegion{T}
end

function Config{T}(ea::ElboArgs,
                   vp::VariationalParams{T},
                   bound_params::VariationalParams{T} = vp[ea.active_sources];
                   loc_width::Float64 = 1e-4,
                   loc_scale::Float64 = 1.0,
                   max_iters::Int = 50,
                   constraints::ConstraintBatch = ConstraintBatch(bound_params, loc_width, loc_scale),
                   free_params::VariationalParams{T} = allocate_free_params(bound_params, constraints),
                   derivs::TransformDerivatives = TransformDerivatives(bound_params, free_params),
                   optim_options::Options = custom_optim_options(max_iters=max_iters),
                   trust_region::NewtonTrustRegion = custom_trust_region())
    return Config(vp, bound_params, free_params, constraints, derivs,
                  optim_options, trust_region)
end

function custom_optim_options(; xtol_abs = 1e-7, ftol_rel = 1e-6, max_iters = 50)
    return Optim.Options(x_tol = xtol_abs, f_tol = ftol_rel, g_tol = 1e-8,
                         iterations = max_iters, store_trace = false,
                         show_trace = false, extended_trace = false)
end

function custom_trust_region(; initial_delta = 10.0, delta_hat = 1e9)
    return Optim.NewtonTrustRegion(initial_delta = initial_delta,
                                   delta_hat = delta_hat)
end

#############
# Objective #
#############

immutable Objective{N,T} <: Function
    ea::ElboArgs
    previous_x::Vector{T}
    sf_free::SensitiveFloat{T}
    cfg::Config{N,T}
end

immutable Gradient{F<:Objective} <: Function
    f::F
end

immutable Hessian{F<:Objective} <: Function
    f::F
end

function Objective(ea::ElboArgs, cfg::Config, x::Vector)
    sf_free = SensitiveFloat{Float64}(length(cfg.free_params[1]), length(cfg.bound_params), true, true)
    previous_x = fill(NaN, length(x))
    return Objective(ea, previous_x, sf_free, cfg)
end

function evaluate!(f::Objective, x::Vector)
    if x != f.previous_x
        copy!(f.previous_x, x)
        from_vector!(f.cfg.free_params, x)
        to_bound!(f.cfg.bound_params, f.cfg.free_params, f.cfg.constraints)
        sf_bound = elbo(f.ea, f.cfg.vp)
        propagate_derivatives!(to_bound!, sf_bound, f.sf_free, f.cfg.free_params,
                               f.cfg.constraints, f.cfg.derivs)
    end
    return nothing
end

function (f::Objective)(x::Vector)
    evaluate!(f, x)
    return -(f.sf_free.v[])
end

function (f::Gradient)(x::Vector, result::Vector)
    evaluate!(f.f, x)
    free_gradient = f.f.sf_free.d
    for i in eachindex(result)
        result[i] = -(free_gradient[i])
    end
    return result
end

function (f::Hessian)(x::Vector, result::Matrix)
    evaluate!(f.f, x)
    free_hessian = f.f.sf_free.h
    for i in eachindex(result)
        result[i] = -(free_hessian[i])
    end
    return result
end

#############
# maximize! #
#############

function maximize!(ea::ElboArgs, vp::VariationalParams{Float64}, cfg::Config = Config(ea, vp))
    enforce!(cfg.bound_params, cfg.constraints)
    to_free!(cfg.free_params, cfg.bound_params, cfg.constraints)
    x = to_vector(cfg.free_params)
    objective = Objective(ea, cfg, x)
    R = Optim.MultivariateOptimizationResults{Float64,1,Optim.NewtonTrustRegion{Float64}}
    result::R = Optim.optimize(objective, Gradient(objective), Hessian(objective),
                               x, cfg.trust_region, cfg.optim_options)
    min_value::Float64 = -(Optim.minimum(result))
    min_solution::Vector{Float64} = Optim.minimizer(result)
    from_vector!(cfg.free_params, min_solution)
    to_bound!(cfg.bound_params, cfg.free_params, cfg.constraints)
    return result.f_calls, min_value, min_solution, result
end

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

end # module
