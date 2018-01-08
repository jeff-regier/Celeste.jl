module ElboMaximize

using Optim
using Optim: Options, NewtonTrustRegion
using ..Model
using ..SensitiveFloats
using ..DeterministicVI
using ..DeterministicVI: init_thread_pool!, ElboIntermediateVariables
using ..DeterministicVI.ConstraintTransforms: TransformDerivatives,
                                              VariationalParams,
                                              ConstraintBatch,
                                              ParameterConstraint,
                                              BoxConstraint,
                                              SimplexConstraint,
                                              allocate_free_params,
                                              to_free!,
                                              to_bound!,
                                              enforce!,
                                              propagate_derivatives!

##################################################
# defaults for optional arguments to `maximize!` #
##################################################

struct ElboConfig{N,T}
    bound_params::VariationalParams{T}
    free_params::VariationalParams{T}
    free_initial_input::Vector{T}
    free_previous_input::Vector{T}
    free_result::SensitiveFloat{T}
    bvn_bundle::Model.BvnBundle{T}
    constraints::ConstraintBatch
    derivs::TransformDerivatives{N,T}
    optim_options::Options
    trust_region::NewtonTrustRegion{T}
end

function ElboConfig(
    ea::ElboArgs,
    vp::VariationalParams{T},
    bound_params::VariationalParams{T} = vp[ea.active_sources];
    termination_callback = nothing,
    loc_width::Float64 = 1e-4,
    loc_scale::Float64 = 1.0,
    max_iters::Int = 50,
    constraints::ConstraintBatch = elbo_constraints(bound_params, loc_width, loc_scale),
    optim_options::Options = elbo_optim_options(max_iters=max_iters,
                                                termination_callback=termination_callback),
    trust_region::NewtonTrustRegion = elbo_trust_region()) where {T}

    free_params = allocate_free_params(bound_params, constraints)
    free_initial_input = to_flat_vector(free_params)
    free_previous_input = similar(free_initial_input)
    free_result = SensitiveFloat{Float64}(length(free_params[1]), length(bound_params), true, true)
    bvn_bundle = Model.BvnBundle{T}(ea.psf_K, ea.S)
    derivs = TransformDerivatives(bound_params, free_params)

    return ElboConfig(bound_params, free_params, free_initial_input,
                      free_previous_input, free_result, bvn_bundle,
                      constraints, derivs, optim_options, trust_region)
end

function elbo_constraints(bound::VariationalParams{T},
                          loc_width::Real = 1.0e-4,
                          loc_scale::Real = 1.0) where {T}
    n_sources = length(bound)
    boxes = Vector{Vector{ParameterConstraint{BoxConstraint}}}(n_sources)
    simplexes = Vector{Vector{ParameterConstraint{SimplexConstraint}}}(n_sources)
    for src in 1:n_sources
        i1, i2 = ids.pos[1], ids.pos[2]
        u1, u2 = bound[src][i1], bound[src][i2]
        boxes[src] = [
            ParameterConstraint(BoxConstraint(u1 - loc_width, u1 + loc_width, loc_scale), i1),
            ParameterConstraint(BoxConstraint(u2 - loc_width, u2 + loc_width, loc_scale), i2),
            ParameterConstraint(BoxConstraint(1e-2, 0.99, 1.0), ids.gal_frac_dev),
            ParameterConstraint(BoxConstraint(1e-2, 0.99, 1.0), ids.gal_axis_ratio),
            ParameterConstraint(BoxConstraint(-10.0, 10.0, 1.0), ids.gal_angle),
            ParameterConstraint(BoxConstraint(0.10, 70.0, 1.0), ids.gal_radius_px),
            ParameterConstraint(BoxConstraint(-1.0, 10.0, 1.0), ids.flux_loc),
            ParameterConstraint(BoxConstraint(1e-4, 0.10, 1.0), ids.flux_scale),
            ParameterConstraint(BoxConstraint(-10.0, 10.0, 1.0), ids.color_mean[:, 1]),
            ParameterConstraint(BoxConstraint(-10.0, 10.0, 1.0), ids.color_mean[:, 2]),
            ParameterConstraint(BoxConstraint(1e-4, 1.0, 1.0), ids.color_var[:, 1]),
            ParameterConstraint(BoxConstraint(1e-4, 1.0, 1.0), ids.color_var[:, 2])
        ]
        simplexes[src] = [
            ParameterConstraint(SimplexConstraint(0.005, 1.0, 2), ids.is_star),
            ParameterConstraint(SimplexConstraint(0.01/NUM_COLOR_COMPONENTS, 1.0, NUM_COLOR_COMPONENTS), ids.k[:, 1]),
            ParameterConstraint(SimplexConstraint(0.01/NUM_COLOR_COMPONENTS, 1.0, NUM_COLOR_COMPONENTS), ids.k[:, 2])
        ]
    end
    return ConstraintBatch(boxes, simplexes)
end

function elbo_optim_options(args...; xtol_abs = 1e-7, ftol_rel = 1e-6, max_iters = 50,
                            termination_callback = nothing, kwargs...)
    return Optim.Options(args...; x_tol = xtol_abs, f_tol = ftol_rel,
                         g_tol = 1e-8, iterations = max_iters,
                         store_trace = false, show_trace = false,
                         extended_trace = false,
                         callback = termination_callback,
                         kwargs...)
end

function elbo_trust_region(; initial_delta = 1.0, delta_hat = 1e9)
    return Optim.NewtonTrustRegion(initial_delta = initial_delta,
                                   delta_hat = delta_hat)
end

function enforce_references!(ea::ElboArgs, vp::VariationalParams{T}, cfg::ElboConfig) where {T}
    @assert length(cfg.bound_params) == length(ea.active_sources)
    for i in 1:length(ea.active_sources)
        cfg.bound_params[i] = vp[ea.active_sources[i]]
    end
    return nothing
end

to_flat_vector(sources::VariationalParams{T}) where {T} = vcat(sources...)::Vector{T}

function to_flat_vector!(x::Vector{T}, sources::VariationalParams{T}) where {T}
    i = 1
    for src in sources
        for j in eachindex(src)
            x[i] = src[j]
            i += 1
        end
    end
    return sources
end

function to_variational_params!(sources::VariationalParams{T}, x::Vector{T}) where {T}
    i = 1
    for src in sources
        for j in eachindex(src)
            src[j] = x[i]
            i += 1
        end
    end
    return sources
end

##########################################
# Preallocated ElboIntermediateVariables #
##########################################

const ELBO_VARS_POOL = Vector{ElboIntermediateVariables{Float64}}()

get_elbo_vars() = ELBO_VARS_POOL[Base.Threads.threadid()]

function __init__()
    init_thread_pool!(ELBO_VARS_POOL, () -> ElboIntermediateVariables(Float64, 1, true, true))
end

# explicitly call this for use with compiled system image
__init__()

##################################
# Callable Types Passed to Optim #
##################################

function evaluate!(ea::ElboArgs, vp::VariationalParams{T}, cfg::ElboConfig, x::Vector) where {T}
    if x != cfg.free_previous_input
        copy!(cfg.free_previous_input, x)
        to_variational_params!(cfg.free_params, x)
        to_bound!(cfg.bound_params, cfg.free_params, cfg.constraints)
        bound_result = elbo(ea, vp, get_elbo_vars(), cfg.bvn_bundle)
        propagate_derivatives!(to_bound!, bound_result,
                               cfg.free_result, cfg.free_params,
                               cfg.constraints, cfg.derivs)
    end
    return nothing
end

# Objective #
#-----------#

struct Objective{N,T} <: Function
    ea::ElboArgs
    vp::VariationalParams{T}
    cfg::ElboConfig{N,T}
end

function (f::Objective)(x::Vector)
    evaluate!(f.ea, f.vp, f.cfg, x)
    return -(f.cfg.free_result.v[])
end

# Gradient #
#----------#

struct Gradient{N,T} <: Function
    ea::ElboArgs
    vp::VariationalParams{T}
    cfg::ElboConfig{N,T}
end

function (f::Gradient)(x::Vector, result::Vector)
    evaluate!(f.ea, f.vp, f.cfg, x)
    free_gradient = f.cfg.free_result.d
    for i in eachindex(result)
        result[i] = -(free_gradient[i])
    end
    return result
end

# Hessian #
#---------#

struct Hessian{N,T} <: Function
    ea::ElboArgs
    vp::VariationalParams{T}
    cfg::ElboConfig{N,T}
end

function (f::Hessian)(x::Vector, result::Matrix)
    evaluate!(f.ea, f.vp, f.cfg, x)
    free_hessian = f.cfg.free_result.h
    for i in eachindex(result)
        result[i] = -(free_hessian[i])
    end
    return result
end

#############
# maximize! #
#############

function maximize!(ea::ElboArgs, vp::VariationalParams{Float64}, cfg::ElboConfig = ElboConfig(ea, vp))
    enforce_references!(ea, vp, cfg)
    enforce!(cfg.bound_params, cfg.constraints)
    to_free!(cfg.free_params, cfg.bound_params, cfg.constraints)
    to_flat_vector!(cfg.free_initial_input, cfg.free_params)
    fill!(cfg.free_previous_input, NaN)
    R = Optim.MultivariateOptimizationResults{Float64,1,Optim.NewtonTrustRegion{Float64}}
    result::R = Optim.optimize(Objective(ea, vp, cfg), Gradient(ea, vp, cfg), Hessian(ea, vp, cfg),
                               cfg.free_initial_input, cfg.trust_region, cfg.optim_options)
    min_value::Float64 = -(Optim.minimum(result))
    min_solution::Vector{Float64} = Optim.minimizer(result)
    to_variational_params!(cfg.free_params, min_solution)
    to_bound!(cfg.bound_params, cfg.free_params, cfg.constraints)
    return result.f_calls, min_value, min_solution, result
end

end # module
