
using OptimizeElbo
import Transform
import Optim

# Put this in OptimizeElbo once it's working.
function maximize_f_newton(f::Function, mp::ModelParams, transform::Transform.DataTransform;
                           omitted_ids=Int64[], xtol_rel = 1e-7, ftol_abs = 1e-6, verbose=false, hess_reg=2.0, max_iters=100)

    kept_ids = setdiff(1:length(UnconstrainedParams), omitted_ids)
    x0 = transform.vp_to_vector(mp.vp, omitted_ids)

    optim_obj_wrap = OptimizeElbo.ObjectiveWrapperFunctions(f, mp, transform, kept_ids, omitted_ids);
    optim_obj_wrap.state.scale = -1.0 # For minimization, which is required by the linesearch algorithm.
    optim_obj_wrap.state.verbose = verbose

    function f_hess_reg!(x, new_hess)
        hess = optim_obj_wrap.f_ad_hessian(x)
        hess_ev = eig(hess)[1]
        min_ev = minimum(hess_ev)
        max_ev = maximum(hess_ev)

        # Make it positive definite.
        if min_ev < 0
            verbose && println("Hessian is negative definite (min eigenvalue $(min_ev)).  Regularizing with $(hess_reg).")
            hess += eye(length(x)) * abs(min_ev) * hess_reg
        end

        new_hess[:,:] = hess
    end

    x0 = transform.vp_to_vector(mp_original.vp, omitted_ids);

    # TODO: are xtol_rel and ftol_abs still good names?
    nm_result = Optim.optimize(optim_obj_wrap.f_value, optim_obj_wrap.f_grad!, f_hess_reg!,
                               x0, method=:newton, iterations=max_iters, xtol=xtol_rel, ftol=ftol_abs)

    iter_count = optim_obj_wrap.state.f_evals
    transform.vector_to_vp!(nm_result.minimum, mp.vp, omitted_ids);
    max_f = -1.0 * nm_result.f_minimum
    max_x = nm_result.minimum

    println("got $max_f at $max_x after $iter_count function evaluations ($(nm_result.iterations) Newton steps)\n")
    iter_count, max_f, max_x, nm_result
end

