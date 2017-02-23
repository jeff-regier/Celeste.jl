module NewtonMaximize

using Optim
using Optim: Options, NewtonTrustRegion
using ..Model
using ..SensitiveFloats
using ..DeterministicVI: ElboArgs
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
                   loc_width::Float64 = 1e-4,
                   loc_scale::Float64 = 1.0,
                   verbose::Bool = false,
                   max_iters::Int = 50,
                   constraints::ConstraintBatch = ConstraintBatch(bound_params, loc_width, loc_scale),
                   free_params::ParameterBatch{T} = allocate_free_params(bound_params, constraints),
                   derivs::TransformDerivatives = TransformDerivatives(bound_params, free_params),
                   optim_options::Options = custom_optim_options(verbose=verbose, max_iters=max_iters),
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

#######################
# maximize_two_steps! #
#######################

function star_only_config(ea::ElboArgs; loc_width = 1.0e-4, loc_scale = 1.0, kwargs...)
    bound = ea.vp[ea.active_sources]
    n_sources = length(bound)
    boxes = Vector{Vector{ParameterConstraint{BoxConstraint}}}(n_sources)
    simplexes = Vector{Vector{ParameterConstraint{SimplexConstraint}}}(n_sources)
    for src in 1:n_sources
        u1, u2 = u_ParameterConstraints(bound[src], loc_width, loc_scale)
        boxes[src] = [
            u1,
            u2,
            ParameterConstraint(BoxConstraint(-1.0, 10.0, 1.0), ids.r1[1]),
            ParameterConstraint(BoxConstraint(1e-4, 0.10, 1.0), ids.r2[1]),
            ParameterConstraint(BoxConstraint(-10.0, 10.0, 1.0), ids.c1[:, 1]),
            ParameterConstraint(BoxConstraint(1e-4, 1.0, 1.0), ids.c2[:, 1])
        ]
        simplexes[src] = [
            ParameterConstraint(SimplexConstraint(0.01/D, 1.0, D), ids.k[:, 1])
        ]
    end
    return Config(ea; constraints = ConstraintBatch(boxes, simplexes), kwargs...)
end

function maximize_two_steps!{F,T}(f::F, ea::ElboArgs{T},
                                  cfg_star::Config = star_only_config(ea),
                                  cfg_both::Config = Config(ea))
    ea.vp[1][ids.a] = [1, 0]
    ea.active_source_star_only = true
    f_evals_star = first(maximize!(f, ea, cfg_star))
    ea.vp[1][ids.a] = [0.8, 0.2]
    ea.active_source_star_only = false
    f_evals_both, max_f_both, max_x_both, nm_result_both = maximize!(f, ea, cfg_both)
    return f_evals_star + f_evals_both, max_f_both, max_x_both, nm_result_both
end

end # module
