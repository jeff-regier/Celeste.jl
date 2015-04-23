# written by Jeffrey Regier
# jeff [at] stat [dot] berkeley [dot] edu

module OptimizeElbo

using NLopt
using CelesteTypes
using Constrain

import ElboDeriv

function get_nlopt_bounds(vs::Vector{Float64})
    # Note that sources are not allowed to move more than
    # one pixel from their starting position in order to
    # avoid label switiching.  (This is why this function gets
    # the variational parameters as an argument.)
    # vs: parameters for a particular source.
    # vp: complete collection of sources.
    lb = Array(Float64, length(all_params))
    lb[ids.chi] = 1e-2
    lb[ids.mu] = vs[ids.mu] - 1.
    [lb[id] = 1e-4 for id in ids.gamma]
    [lb[id] = 1e-4 for id in ids.zeta]
    [lb[id] = 1e-4 for id in ids.kappa]
    [lb[id] = -10. for id in ids.beta]
    [lb[id] = 1e-4 for id in ids.lambda]
    lb[ids.theta] = 1e-2
    lb[ids.rho] = 1e-4
    lb[ids.phi] = -1e10 #-pi/2 + 1e-4
    lb[ids.sigma] = 0.2

    ub = Array(Float64, length(all_params))
    ub[ids.chi] = 1 - 1e-2
    ub[ids.mu] = vs[ids.mu] + 1.
    [ub[id] = 1e12 for id in ids.gamma]
    [ub[id] = 1e-1 for id in ids.zeta]
    [ub[id] = 1 - 1e-4 for id in ids.kappa]
    [ub[id] = 10. for id in ids.beta]
    [ub[id] = 1. for id in ids.lambda]
    ub[ids.theta] = 1 - 1e-2
    ub[ids.rho] = 1. - 1e-4
    ub[ids.phi] = 1e10 #pi/2 - 1e-4
    ub[ids.sigma] = 15.

    lb, ub
end


function get_nlopt_unconstrained_bounds(vp::Vector{Vector{Float64}},
                                        omitted_ids,
                                        transform::DataTransform)
    # Note that sources are not allowed to move more than
    # one pixel from their starting position in order to
    # avoid label switiching.  (This is why this function gets
    # the variational parameters as an argument.)
    #
    # vp: _Constrained_ variational parameters.
    # omitted_ids: Ids of _unconstrained_ variational parameters to be omitted.
    # unconstrain_fn: A function to convert VariationalParameters to a
    #   vector that can be passed to the optimizer.

    lbs = [ get_nlopt_bounds(vs)[1] for vs in vp]
    ubs = [ get_nlopt_bounds(vs)[2] for vs in vp]
    transform.vp_to_vector(lbs, omitted_ids), transform.vp_to_vector(ubs, omitted_ids)
end


function print_params(vp)
    for vs in vp
        for n in names(ids)
            println(n, ": ", vs[ids.(n)])
        end
        println("-----------------\n")
    end
end


function maximize_f(f::Function, blob::Blob, mp::ModelParams, transform::DataTransform;
    omitted_ids=Int64[], xtol_rel = 1e-7, ftol_abs = 1e-6)
    # Maximize using NLOpt and unconstrained coordinates.
    #
    # Args:
    #   - f: A function that takes a blob and constrianed coordinates (e.g. ElboDeriv.elbo)
    #   - blob: Input for f
    #   - mp: Constrained initial ModelParams
    #   - omitted_ids: Omitted ids from the _unconstrained_ parameterization (i.e. elements
    #       of free_ids).

    x0 = transform.vp_to_vector(mp.vp, omitted_ids)
    iter_count = 0

    function objective_and_grad(x::Vector{Float64}, g::Vector{Float64})
        println("Iter: ", iter_count)
        # Evaluate in the constrained space and then unconstrain again.
        transform.vector_to_vp!(x, mp.vp, omitted_ids)
        elbo = f(blob, mp)
        elbo_trans = transform.transform_sensitive_float(elbo, mp)
        if length(g) > 0
            svs = [elbo2.d[:, s] for s in 1:mp.S]
            g[:] = reduce(vcat, svs)
        end

        iter_count += 1
        print_params(mp.vp)
        println("elbo: ", elbo.v)
        println("xtol_rel: $xtol_rel ;  ftol_abs: $ftol_abs")
        println("rescaling: ", rescaling)
        println("\n=======================================\n")
        elbo.v
    end

    opt = Opt(:LD_LBFGS, length(x0))
    lbs, ubs = get_nlopt_unconstrained_bounds(mp.vp, omitted_ids, transform)
    for i in 1:length(x0)
        if !(lbs[i] <= x0[i] <= ubs[i])
            println("coordinate $i falsity: $(lbs[i]) <= $(x0[i]) <= $(ubs[i])")
        end
    end
    lower_bounds!(opt, lbs)
    upper_bounds!(opt, ubs)
    max_objective!(opt, objective_and_grad)
    xtol_rel!(opt, xtol_rel)
    ftol_abs!(opt, ftol_abs)
    (max_f, max_x, ret) = optimize(opt, x0)

    println("got $max_f at $max_x after $iter_count iterations (returned $ret)\n")
    iter_count, max_f, max_x, ret
end


function maximize_f(f::Function, blob::Blob, mp::ModelParams;
    omitted_ids=Int64[], xtol_rel = 1e-7, ftol_abs = 1e-6)
    # Default to the rectangular transform
    maximize_f(f, blob, mp, rect_transform, omitted_ids, xtol_rel, ftol_abs)
end

function maximize_elbo(blob::Blob, mp::ModelParams)
    omitted_ids = setdiff(all_params, [ids.gamma, ids.zeta, ids.kappa[:], ids.beta[:]])
    maximize_f(ElboDeriv.elbo, blob, mp, omitted_ids=omitted_ids)

    maximize_f(ElboDeriv.elbo, blob, mp, ftol_abs=1e-6)
end


function maximize_likelihood(blob::Blob, mp::ModelParams)
    omitted_ids = [ids.kappa[:], ids.lambda[:], ids.zeta]
    maximize_f(ElboDeriv.elbo_likelihood, blob, mp, omitted_ids=omitted_ids)
end

end
