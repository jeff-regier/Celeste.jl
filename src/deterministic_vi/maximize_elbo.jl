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

    f_evals = 0
    const print_every_n = 10

    function f_wrapped_nocache(x::Vector{Float64})
        f_evals += 1

        # Evaluate in the constrained space and then unconstrain again.
        transform.array_to_vp!(reshape(x, x_size), ea.vp, omitted_ids)
        f_res = f(ea)

        # TODO: Add an option to print either the transformed or
        # free parameterizations.

        if verbose || (f_evals % print_every_n == 0)
            Log.info("f_evals: $(f_evals) value: $(f_res.v[1])")
        end

        if verbose
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
                state_df[Symbol(string("grad", s))] = f_res.d[:, s]
            end

            Log.info(repr(state_df))
            Log.info("=======================================\n")
        end

        transform.transform_sensitive_float(f_res, ea.vp, ea.active_sources)
    end

    last_sf = zero_sensitive_float(UnconstrainedParams, Sa)
    last_x = [ NaN ]

    function f_wrapped_cached(x::Vector{Float64})
        if x != last_x
            last_x = deepcopy(x)
            last_sf = deepcopy(f_wrapped_nocache(x))
        end

        last_sf
    end

    function neg_f_value{T <: Number}(x::Vector{T})
        @assert length(x) == x_length
        -f_wrapped_cached(x).v[1]
    end

    function neg_f_grad!{T <: Number}(x::Vector{T}, grad::Vector{T})
        @assert length(x) == x_length
        res = f_wrapped_cached(x)
        if length(grad) > 0
            svs = [res.d[kept_ids, si] for si in 1:transform.active_S]
            grad[:] = reduce(vcat, svs)
        end
        grad .*= -1
    end

    function neg_f_hessian!{T <: Number}(x::Vector{T}, hess::Matrix{T})
        @assert length(x) == x_length
        res = f_wrapped_cached(x)
        all_kept_ids = Int[]
        for sa=1:transform.active_S
            append!(all_kept_ids, kept_ids + (sa - 1) * length(kept_ids))
        end
        sub_hess = res.h[all_kept_ids, all_kept_ids]
        hess[:, :] = -1 .* 0.5 * (sub_hess + sub_hess')
    end

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

    nm_result = Optim.optimize(neg_f_value,
                               neg_f_grad!,
                               neg_f_hessian!,
                               x0[:],
                               tr_method,
                               options)

    transform.array_to_vp!(reshape(nm_result.minimum, size(x0)),
                           ea.vp, omitted_ids)
    max_f = -1.0 * nm_result.f_minimum
    max_x = nm_result.minimum

    Log.info(string("got $max_f at $max_x after $f_evals function evaluations ",
            "($(nm_result.iterations) Newton steps)\n"))
    f_evals, max_f, max_x, nm_result
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


