
using OptimizeElbo
using Transform

# Put this in OptimizeElbo once it's working.
function maximize_f_newton(f::Function, blob::Blob, mp::ModelParams, transform::DataTransform,
                           lbs::Array{Float64, 1}, ubs::Array{Float64, 1};
                           omitted_ids=Int64[], xtol_rel = 1e-7, ftol_abs = 1e-6)

    kept_ids = setdiff(1:length(UnconstrainedParams), omitted_ids)
    x0 = transform.vp_to_vector(mp.vp, omitted_ids)

    obj_wrapper = ObjectiveWrapperFunctions(mp -> f(blob, mp), mp, transform, kept_ids, omitted_ids);

    opt = Opt(:LD_LBFGS, length(x0))
    for i in 1:length(x0)
        if !(lbs[i] <= x0[i] <= ubs[i])
            println("coordinate $i falsity: $(lbs[i]) <= $(x0[i]) <= $(ubs[i])")
        end
    end
    lower_bounds!(opt, lbs)
    upper_bounds!(opt, ubs)
    max_objective!(opt, obj_wrapper.f_value_grad!)
    xtol_rel!(opt, xtol_rel)
    ftol_abs!(opt, ftol_abs)
    (max_f, max_x, ret) = optimize(opt, x0)

    iter_count = obj_wrapper.state.f_evals
    println("got $max_f at $max_x after $iter_count iterations (returned $ret)\n")
    iter_count, max_f, max_x, ret
end

