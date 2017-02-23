module NewtonMaximize

using Optim
using Optim: Options, NewtonTrustRegion
using ..Model
using ..SensitiveFloats
using ..DeterministicVI: ElboArgs
using ...ConstraintTransforms: TransformDerivatives,
                               ParameterBatch,
                               ConstraintBatch,
                               allocate_free_params,
                               to_free!,
                               to_bound!,
                               enforce!,
                               propagate_derivatives!

##################################################
# defaults for optional arguments to `maximize!` #
##################################################

immutable Config{N,T}
    bound_params::ParameterBatch{T}
    free_params::ParameterBatch{T}
    constraints::ConstraintBatch
    derivs::TransformDerivatives{N,T}
    optim_options::Options{Void}
    trust_region::NewtonTrustRegion{T}
    verbose::Bool
end

function Config{T}(bound_params::ParameterBatch{T};
                   constraints::ConstraintBatch = ConstraintBatch(bound_params),
                   free_params::ParameterBatch{T} = allocate_free_params(bound_params, constraints),
                   derivs::TransformDerivatives = TransformDerivatives(bound_params, free_params),
                   verbose::Bool = false,
                   optim_options::Options = custom_optim_options(verbose=verbose),
                   trust_region::NewtonTrustRegion = custom_trust_region())
    return Config(bound_params, free_params, constraints, derivs,
                  optim_options, trust_region, verbose)
end

Config(ea::ElboArgs; kwargs...) = Config(ea.vp[ea.active_sources]; kwargs...)

function custom_optim_options(; xtol_abs = 1e-7, ftol_rel = 1e-6, max_iters = 50, verbose = false)
    return Optim.Options(x_tol = xtol_abs, f_tol = ftol_rel, g_tol = 1e-8,
                         iterations = max_iters, store_trace = verbose,
                         show_trace = false, extended_trace = verbose)
end

function custom_trust_region(; initial_delta = 10.0, delta_hat = 1e9, eta = 0.1,
                             rho_lower = 0.25, rho_upper = 0.75)
    return Optim.NewtonTrustRegion(initial_delta = initial_delta,
                                   delta_hat = delta_hat,
                                   eta = eta,
                                   rho_lower = rho_lower,
                                   rho_upper = rho_upper)
end

#############
# Objective #
#############

immutable Objective{F,N,T} <: Function
    f::F
    ea::ElboArgs{T}
    f_evals::Ref{Int}
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

function Objective{F,T}(f::F, ea::ElboArgs{T}, cfg::Config, x::Vector)
    sf_free = SensitiveFloat{T}(length(cfg.free_params[1]), length(cfg.bound_params), true, true)
    previous_x = fill(NaN, length(x))
    return Objective(f, ea, Ref{Int}(0), previous_x, sf_free, cfg)
end

function evaluate!(f::Objective, x::Vector)
    if x != f.previous_x
        copy!(f.previous_x, x)
        from_vector!(f.cfg.free_params, x)
        to_bound!(f.cfg.bound_params, f.cfg.free_params, f.cfg.constraints)
        sf_bound = f.f(f.ea)
        f.f_evals[] += 1
        f.cfg.verbose && log_eval(f, sf_bound)
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

function log_eval(f::Objective, sf)
    Log.debug("f_evals=$(f.f_evals[]) | value=$(sf.v[])")
    Log.debug(sprint(io -> show_params(io, f.cfg.bound_params, sf)))
    Log.debug("=======================================\n")
end

function show_params(io, sources, sf)
    n_params = size(sf.d, 1)
    param_names = n_params == length(ids_names) ? ids_names : ["x$i" for i=1:n_params]
    for src in 1:length(sources)
        println(io, "source $(src):")
        for i in 1:n_params
            println(io, "  $(param_names[i])=$(sources[src][i]), ∂$(param_names[i])=$(sf.d[i, src])")
        end
    end
end

#############
# maximize! #
#############

function maximize!{F,T}(f::F, ea::ElboArgs{T}, cfg::Config = Config(ea))
    enforce!(cfg.bound_params, cfg.constraints)
    to_free!(cfg.free_params, cfg.bound_params, cfg.constraints)
    x = to_vector(cfg.free_params)
    objective = Objective(f, ea, cfg, x)
    R = Optim.MultivariateOptimizationResults{T,1,Optim.NewtonTrustRegion{T}}
    result::R = Optim.optimize(objective, Gradient(objective), Hessian(objective),
                               x, cfg.trust_region, cfg.optim_options)
    min_value::T = -(Optim.minimum(result))
    min_solution::Vector{T} = Optim.minimizer(result)
    from_vector!(cfg.free_params, min_solution)
    to_bound!(cfg.bound_params, cfg.free_params, cfg.constraints)
    return objective.f_evals[]::Int64, min_value, min_solution, result
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
