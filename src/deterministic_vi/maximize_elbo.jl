# The main reason we need this is to have a mutable type to keep
# track of function evaluations, but we can keep other metadata
# in it as well.
type WrapperState
    f_evals::Int
    verbose::Bool
    print_every_n::Int
    scale::Float64
end


function print_status{T <: Number}(
                    state::WrapperState,
                    iter_vp::VariationalParams{T},
                    value::T,
                    grad::Array{T})
    if state.verbose || (state.f_evals % state.print_every_n == 0)
        Log.info("f_evals: $(state.f_evals) value: $(value)")
    end
    if state.verbose
        S = length(iter_vp)
        if length(iter_vp[1]) == length(ids_names)
        state_df = DataFrames.DataFrame(names=ids_names)
        elseif length(iter_vp[1]) == length(ids_free_names)
        state_df = DataFrames.DataFrame(names=ids_free_names)
        else
        state_df = DataFrames.DataFrame(
        names=[ "x$i" for i=1:length(iter_vp[1, :])])
        end
        for s=1:S
        state_df[Symbol(string("val", s))] = iter_vp[s]
        end
        for s=1:S
        state_df[Symbol(string("grad", s))] = grad[:, s]
        end
        Log.info(repr(state_df))
        Log.info("=======================================\n")
    end
end

"""
Optimizes f using Newton's method and exact Hessians.  For now, it is
not clear whether this or BFGS is better, so it is kept as a separate function.

Args:
  - f: A function that takes elbo args and constrained coordinates
       (e.g. DeterministicVI.elbo)
  - ea: Constrained initial ElboArgs
  - lbs: An array of lower bounds (in the transformed space)
  - ubs: An array of upper bounds (in the transformed space)
  - omitted_ids: Omitted ids from the _unconstrained_ parameterization
      (i.e. elements of free_ids).
  - xtol_rel: X convergence
  - ftol_abs: F convergence
  - verbose: Print detailed output

Returns:
  - iter_count: The number of iterations taken
  - max_f: The maximum function value achieved
  - max_x: The optimal function input
  - ret: The return code of optimize()
"""
function maximize_f(f::Function,
                    ea::ElboArgs,
                    transform::DataTransform;
                    omitted_ids=Int[],
                    xtol_rel=1e-7,
                    ftol_abs=1e-6,
                    verbose=false,
                    max_iters=50,
                    fast_hessian=true)
    # Make sure the model parameters are within the transform bounds
    enforce_bounds!(ea.vp, ea.active_sources, transform)
    @assert ea.active_sources == transform.active_sources

    kept_ids = setdiff(1:length(UnconstrainedParams), omitted_ids)
    x_length = length(kept_ids) * transform.active_S
    x_size = (length(kept_ids), transform.active_S)

    Sa = length(ea.active_sources)

    state = WrapperState(0, false, 10, 1.0)

    function f_objective_nocache(x::Vector{Float64})
        state.f_evals += 1
        # Evaluate in the constrained space and then unconstrain again.
        transform.array_to_vp!(reshape(x, x_size), ea.vp, omitted_ids)
        f_res = f(ea)

        # TODO: Add an option to print either the transformed or
        # free parameterizations.
        print_status(state, ea.vp[ea.active_sources], f_res.v[1], f_res.d)
        transform.transform_sensitive_float(f_res, ea.vp, ea.active_sources)
    end

    last_sf = zero_sensitive_float(UnconstrainedParams, Sa)
    last_x = [ NaN ]

    function f_objective_cached(x::Vector{Float64})
        if x != last_x
            last_x = deepcopy(x)
            last_sf = deepcopy(f_objective_nocache(x))
        end

        last_sf
    end

    function f_value_grad{T <: Number}(x::Vector{T})
        @assert length(x) == x_length
        res = f_objective_cached(x)
        grad = zeros(T, length(x))
        if length(grad) > 0
            svs = [res.d[kept_ids, si] for si in 1:transform.active_S]
            grad[:] = reduce(vcat, svs)
        end
        state.scale * res.v[1], state.scale .* grad
    end

    # The remaining functions are scaled and take matrices.
    function f_value_grad!{T <: Number}(x::Vector{T}, grad::Vector{T})
        @assert length(x) == x_length
        @assert length(x) == length(grad)
        value, grad[:,:] = f_value_grad(x)
        value
    end

    function f_value{T <: Number}(x::Vector{T})
        @assert length(x) == x_length
        f_value_grad(x)[1]
    end

    function f_grad!{T <: Number}(x::Vector{T}, grad::Vector{T})
        @assert length(x) == x_length
        grad[:,:] = f_value_grad(x)[2]
    end

    function f_hessian{T <: Number}(x::Vector{T})
        @assert length(x) == x_length
        res = f_objective_cached(x)
        all_kept_ids = Int[]
        for sa=1:transform.active_S
            append!(all_kept_ids, kept_ids + (sa - 1) * length(kept_ids))
        end
        sub_hess = res.h[all_kept_ids, all_kept_ids]
        state.scale .* 0.5 * (sub_hess + sub_hess')
    end

    function f_hessian!{T <: Number}(x::Vector{T}, hess::Matrix{T})
        hess[:, :] = f_hessian(x)
    end

    # For minimization, which is required by the linesearch algorithm.
    state.scale = -1.0
    state.verbose = verbose

    x0 = transform.vp_to_array(ea.vp, omitted_ids)
    tr_method = Optim.NewtonTrustRegion(
                    initial_delta=10.0,
                    delta_hat=1e9,
                    eta=0.1,
                    rho_lower=0.25,
                    rho_upper=0.75)

    options = Optim.OptimizationOptions(;
        x_tol = xtol_rel, f_tol = ftol_abs, g_tol = 1e-8,
        iterations = max_iters, store_trace = verbose,
        show_trace = false, extended_trace = verbose)

    nm_result = Optim.optimize(f_value,
                               f_grad!,
                               f_hessian!,
                               x0[:], tr_method, options)

    iter_count = state.f_evals
    transform.array_to_vp!(reshape(nm_result.minimum, size(x0)),
                           ea.vp, omitted_ids)
    max_f = -1.0 * nm_result.f_minimum
    max_x = nm_result.minimum

    Log.info(string("got $max_f at $max_x after $iter_count function evaluations ",
            "($(nm_result.iterations) Newton steps)\n"))
    iter_count, max_f, max_x, nm_result
end


function maximize_f(f::Function,
                    ea::ElboArgs;
                    loc_width=1.5e-3,
                    omitted_ids=Int[],
                    xtol_rel=1e-7,
                    ftol_abs=1e-6,
                    verbose=false,
                    max_iters=50,
                    fast_hessian=true)
    transform = get_mp_transform(ea.vp, ea.active_sources, loc_width=loc_width)

    maximize_f(f, ea, transform;
                omitted_ids=omitted_ids,
                xtol_rel=xtol_rel,
                ftol_abs=ftol_abs,
                verbose=verbose,
                max_iters=max_iters,
                fast_hessian=fast_hessian)
end


