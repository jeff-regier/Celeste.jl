##############
# maximize_f #
##############

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
function maximize_f{F}(f::F, ea::ElboArgs, transform::DataTransform;
                       omitted_ids=Int[],
                       xtol_rel=1e-7,
                       ftol_abs=1e-6,
                       verbose=false,
                       max_iters=50,
                       fast_hessian=true)
    # Make sure the model parameters are within the transform bounds
    enforce_bounds!(ea.vp, ea.active_sources, transform)
    @assert ea.active_sources == transform.active_sources

    f_evals = 0
    print_every_n = 10
    n_active_sources = length(transform.active_sources)
    kept_ids = setdiff(1:length(UnconstrainedParams), omitted_ids)
    all_kept_ids = Int[]
    for i in eachindex(transform.active_sources)
        append!(all_kept_ids, kept_ids + (i - 1) * length(kept_ids))
    end

    x0 = vec(Transform.vp_to_array(transform, ea.vp, omitted_ids))
    last_sf = zero_sensitive_float(UnconstrainedParams, n_active_sources)
    last_x = fill(NaN, size(x0))

    f_wrapped_nocache! = (x::Vector) -> begin
        # Evaluate in the constrained space and then unconstrain again.
        reshaped_x = reshape(x, length(kept_ids), n_active_sources)
        Transform.array_to_vp!(transform, reshaped_x, ea.vp, kept_ids, omitted_ids)
        f_res = f(ea)
        f_evals += 1
        log_f_eval(verbose, f_evals, print_every_n, ea, f_res)
        Transform.transform_sensitive_float!(transform, last_sf, f_res, ea.vp, ea.active_sources)
    end

    f_wrapped_cached! = (x::Vector) -> begin
        if x != last_x
            copy!(last_x, x)
            f_wrapped_nocache!(x)
        end
        return last_sf
    end

    neg_f_value = (x::Vector) -> begin
        sf = f_wrapped_cached!(x)
        return -(sf.v[])
    end

    neg_f_grad! = (x::Vector, grad::Vector) -> begin
        sf = f_wrapped_cached!(x)
        for i in kept_ids, j in 1:n_active_sources
            grad[i, j] = -sf.d[i, j]
        end
        return grad
    end

    neg_f_hessian! = (x::Vector, hess::Matrix) -> begin
        sf = f_wrapped_cached!(x)
        for i in all_kept_ids, j in all_kept_ids
            hess[i, j] = sf.h[i, j]
        end
        Transform.symmetrize!(hess, -0.5)
        return hess
    end

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
                               x0,
                               tr_method,
                               options)

    reshaped_min = reshape(Optim.minimizer(nm_result), size(x0))
    Transform.array_to_vp!(transform, reshaped_min, ea.vp, kept_ids, omitted_ids)
    max_f = -1.0 * Optim.minimum(nm_result)
    max_x = Optim.minimizer(nm_result)

    Log.info("elbo is $max_f after $(nm_result.iterations) Newton steps")
    return f_evals, max_f, max_x, nm_result
end

function maximize_f{F}(f::F, ea::ElboArgs;
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

function log_f_eval(verbose, f_evals, print_every_n, ea::ElboArgs, f_res)
    # TODO: Add an option to print either the transformed or
    # free parameterizations.
    if verbose || (f_evals % print_every_n == 0)
        Log.info("f_evals=$(f_evals) | value=$(f_res.v[])")
    end
    if verbose
        iter_vp = ea.vp[ea.active_sources]

        if length(iter_vp[1]) == length(ids_names)
            state_df = DataFrames.DataFrame(names=ids_names)
        elseif length(iter_vp[1]) == length(ids_free_names)
            state_df = DataFrames.DataFrame(names=ids_free_names)
        else
            state_df = DataFrames.DataFrame(names=[ "x$i" for i=1:length(iter_vp[1, :])])
        end

        for s in eachindex(iter_vp)
            state_df[Symbol(string("val", s))] = iter_vp[s]
        end

        for s in eachindex(iter_vp)
            state_df[Symbol(string("grad", s))] = f_res.d[:, s]
        end

        Log.info(repr(state_df))
        Log.info("=======================================\n")
    end
end
