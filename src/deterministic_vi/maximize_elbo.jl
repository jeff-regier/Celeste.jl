##############
# maximize_f #
##############

"""
Optimizes f using Newton's method and exact Hessians.

Returns:
  - iter_count: The number of iterations taken
  - max_f: The maximum function value achieved
  - max_x: The optimal function input
  - ret: The return code of optimize()
"""
function maximize_f{F}(f::F, ea::ElboArgs, transform::DataTransform;
                       omitted_ids=Int[],
                       use_default_optim_params=false,
                       xtol_abs=1e-7,
                       ftol_rel=1e-6,
                       verbose=false,
                       max_iters=50)
    # Make sure the model parameters are within the transform bounds
    enforce_bounds!(ea.vp, ea.active_sources, transform)
    @assert ea.active_sources == transform.active_sources

    f_evals = 0
    n_active_sources::Int = length(transform.active_sources)
    kept_ids::Vector{Int} = setdiff(1:length(UnconstrainedParams), omitted_ids)

    all_kept_ids::Vector{Int} = Int[]
    for i in eachindex(transform.active_sources)
        append!(all_kept_ids, kept_ids + (i - 1) * length(kept_ids))
    end

    x0 = vec(Transform.vp_to_array(transform, ea.vp, omitted_ids))
    last_sf::SensitiveFloat{Float64} = SensitiveFloat{Float64}(
                    length(UnconstrainedParams), n_active_sources, true, true)
    last_x::Vector{Float64} = fill(NaN, size(x0))

    f_wrapped_nocache! = (x::Vector) -> begin
        # Evaluate in the constrained space and then unconstrain again.
        reshaped_x = reshape(x, length(kept_ids), n_active_sources)
        assert(all(!isnan(reshaped_x)))
        Transform.array_to_vp!(transform, reshaped_x, ea.vp, kept_ids)
        for s in 1:size(ea.vp, 2)
            assert(!any(isnan(ea.vp[s])))
        end 
        f_res = f(ea)
        f_evals += 1
        verbose && record_f_eval(f_evals, ea, f_res)
        Transform.transform_sensitive_float!(transform, last_sf, f_res, ea.vp, ea.active_sources)
    end

    f_wrapped_cached! = (x::Vector) -> begin
        if x != last_x
            copy!(last_x, x)
            f_wrapped_nocache!(x)
        end
        return nothing
    end

    neg_f_value = (x::Vector) -> begin
        f_wrapped_cached!(x)
        return -(last_sf.v[])
    end

    neg_f_grad! = (x::Vector, grad::Vector) -> begin
        f_wrapped_cached!(x)
        for i_free in 1:length(kept_ids)
            i_bnd = kept_ids[i_free]
            for j in 1:n_active_sources
                grad[i_free, j] = -last_sf.d[i_bnd, j]
            end
        end
        return grad
    end

    neg_f_hessian! = (x::Vector, hess::Matrix) -> begin
        f_wrapped_cached!(x)
        for i_free in 1:length(kept_ids)
            i_bnd = kept_ids[i_free]
            for j_free in 1:length(all_kept_ids)
                j_bnd = kept_ids[j_free]
                hess[i_free, j_free] = last_sf.h[i_bnd, j_bnd]
            end
        end
        Transform.symmetrize!(hess, -0.5)
        return hess
    end

    if use_default_optim_params
        tr_method = Optim.NewtonTrustRegion()
    else
        tr_method = Optim.NewtonTrustRegion(
            initial_delta=10.0,
            delta_hat=1e9,
            eta=0.1,
            rho_lower=0.25,
            rho_upper=0.75)
    end

    options = Optim.Options(;
        x_tol = xtol_abs, f_tol = ftol_rel, g_tol = 1e-8,
        iterations = max_iters, store_trace = verbose,
        show_trace = false, extended_trace = verbose)

    nm_result = Optim.optimize(neg_f_value,
                               neg_f_grad!,
                               neg_f_hessian!,
                               x0,
                               tr_method,
                               options)

    reshaped_min = reshape(Optim.minimizer(nm_result), size(x0))
    Transform.array_to_vp!(transform, reshaped_min, ea.vp, kept_ids)
    max_f = -1.0 * Optim.minimum(nm_result)
    max_x = Optim.minimizer(nm_result)

    #Log.debug("elbo is $max_f after $(nm_result.iterations) Newton steps")
    return f_evals, max_f, max_x, nm_result, transform
end

function maximize_f{F}(f::F, ea::ElboArgs;
                       loc_width=1e-4, # about a pixel either direction
                       loc_scale=1.0,
                       omitted_ids=Int[],
                       xtol_abs=1e-7,
                       ftol_rel=1e-6,
                       verbose=false,
                       max_iters=50,
                       use_default_optim_params=false)
    transform = get_mp_transform(ea.vp, ea.active_sources,
                                 loc_width=loc_width, loc_scale=loc_scale)

    maximize_f(f, ea, transform;
                omitted_ids=omitted_ids,
                xtol_abs=xtol_abs,
                ftol_rel=ftol_rel,
                verbose=verbose,
                max_iters=max_iters,
                use_default_optim_params=use_default_optim_params)
end


function maximize_f_two_steps{F}(f::F, ea::ElboArgs;
                                 loc_width=1e-4, # about a pixel either direction
                                 loc_scale=1.0,
                                 omitted_ids=Int[],
                                 xtol_abs=1e-7,
                                 ftol_rel=1e-6,
                                 verbose=false,
                                 max_iters=50,
                                 use_default_optim_params=false)

    transform = get_mp_transform(ea.vp, ea.active_sources,
                                 loc_width=loc_width, loc_scale=loc_scale)

    star_ids_free = vcat(ids_free.u,
                         ids_free.r1[1], ids_free.r2[1],
                         ids_free.c1[:, 1][:], ids_free.c2[:, 1][:],
                         ids_free.k[:, 1][:])

    star_omitted_ids = union(setdiff(1:length(ids_free), star_ids_free),
                             omitted_ids)

     ea.vp[1][ids.a] = [1, 0]
     ea.active_source_star_only = true
     f_evals_star, max_f_star, max_x_star, nm_result_star =
         maximize_f(f, ea, transform;
                    omitted_ids=star_omitted_ids,
                    xtol_abs=xtol_abs,
                    ftol_rel=ftol_rel,
                    verbose=verbose,
                    max_iters=max_iters,
                    use_default_optim_params=use_default_optim_params)

    ea.vp[1][ids.a] = [0.8, 0.2]
    ea.active_source_star_only = false
    f_evals_both, max_f_both, max_x_both, nm_result_both =
        maximize_f(f, ea, transform;
                   omitted_ids=omitted_ids,
                   xtol_abs=xtol_abs,
                   ftol_rel=ftol_rel,
                   verbose=verbose,
                   max_iters=max_iters,
                   use_default_optim_params=use_default_optim_params)

    # It would be nice to somehow return both opimization results.
    f_evals_star + f_evals_both, max_f_both, max_x_both, nm_result_both, transform
end


function record_f_eval(f_evals, ea::ElboArgs, f_res)
    # TODO: Add an option to print either the transformed or
    # free parameterizations.
    Log.debug("f_evals=$(f_evals) | value=$(f_res.v[])")

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

    Log.debug(repr(state_df))
    Log.debug("=======================================\n")
end
