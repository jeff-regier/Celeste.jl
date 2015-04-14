# written by Jeffrey Regier
# jeff [at] stat [dot] berkeley [dot] edu

module OptimizeElbo

using NLopt
using CelesteTypes

import ElboDeriv


const rescaling = ones(length(all_params))

# Rescale some parameters to have similar dimensions to everything else.
[rescaling[id] *= 1e-3 for id in ids.gamma]
#rescaling[ids.chi] *= 1e1

# Rescaling for the unconstrained parameterization.
const rescaling_free = ones(length(all_params_free))
# Perhaps this is not necessary with the unconstrained parametrization?
#[rescaling_free[id] *= 1e-3 for id in ids_free.gamma_free]

function scale_deriv(elbo::SensitiveFloat, omitted_ids, scaling_vector)
    # Move between scaled and unscaled parameterizations.

    left_ids = setdiff(all_params, [omitted_ids, ids.kappa[end, :][:]])
    new_P = length(left_ids)

    elbo_new = zero_sensitive_float(elbo.source_index, left_ids)
    elbo_new.v = elbo.v

    for p1 in 1:length(left_ids)
        p0 = left_ids[p1]
        elbo_new.d[p1, :] = elbo.d[p0, :] ./ scaling_vector[p0]

        for i = 1:2
            if p0 == ids.kappa[1, i]
                for s1 in 1:length(elbo.source_index)
                    elbo_new.d[p1, s1] -= elbo.d[ids.kappa[2, i], s1]
                end
            end
        end
    end

    elbo_new
end


function vp_to_coordinates(vp::Vector{Vector{Float64}}, omitted_ids::Vector{Int64})
    # vp = variational parameters
    # coordinates = for optimizer

    # The last kappa coordinates are excluded because they can be inferred from
    # the other kappa values.
    left_ids = setdiff(all_params, [omitted_ids, ids.kappa[end, :][:]])
    new_P = length(left_ids)

    S = length(vp)
    vp_new = [zeros(new_P) for s in 1:S]

    for p1 in 1:length(left_ids)
        p0 = left_ids[p1]
        [vp_new[s][p1] = vp[s][p0] * rescaling[p0] for s in 1:S]
    end

    reduce(vcat, vp_new)
end


function coordinates_to_vp!(xs::Vector{Float64}, vp::Vector{Vector{Float64}},
        omitted_ids::Vector{Int64})
    left_ids = setdiff(all_params, [omitted_ids, ids.kappa[end, :][:]])

    P = length(left_ids)
    @assert length(xs) % P == 0
    S = int(length(xs) / P)
    xs2 = reshape(xs, P, S)

    for s in 1:S
        for p1 in 1:length(left_ids)
            p0 = left_ids[p1]
            vp[s][p0] = xs2[p1, s] ./ rescaling[p0]
        end
        if ids.kappa[1, 1] in left_ids
            # Each column of kappa is the probability of being a 
            # type of celestial object (currently star or galax).
            # Here, assume that there are only two.
            vp[s][ids.kappa[end, :]] = 1. - vp[s][ids.kappa[1, :]]
        end
    end
end


function vp_to_free_coordinates(vp::Vector{Vector{Float64}}, omitted_ids::Vector{Int64})
    # vp = constrained variational parameters
    # coordinates = unconstrained coordinates for optimizer
    # omitted_ids: ids to be omitted from the _unconstrained_ vb parameters
    #    (i.e. indices from ids_free)

    vp_free = unconstrain_vp(vp)
    left_ids = setdiff(all_params_free,
                       [omitted_ids, ids_free.kappa[end, :][:]])
    new_P = length(left_ids)

    S = length(vp_free)
    vp_new = [zeros(new_P) for s in 1:S]

    for p1 in 1:length(left_ids)
        p0 = left_ids[p1]
        [vp_new[s][p1] = vp_free[s][p0] * rescaling_free[p0] for s in 1:S]
    end

    reduce(vcat, vp_new)
end

function free_coordinates_to_vp!(xs::Vector{Float64}, vp::Vector{Vector{Float64}},
                                 omitted_ids::Vector{Int64})
    # Convert the uncontrained optimization vector to the contrained variational parameters.
    # xs: A vector of the unconstrained parameters for the optimizer.
    # vp: A VariationalParams object to be updated in place with the
    #     constrained parameterization.
    # omitted_ids: ids to be omitted from the _unconstrained_ vb parameters
    #    (i.e. indices from ids_free)

    left_ids = setdiff(all_params_free, [omitted_ids, ids_free.kappa[end, :][:]])

    P = length(left_ids)
    @assert length(xs) % P == 0
    S = int(length(xs) / P)
    xs2 = reshape(xs, P, S)

    vp_free = unconstrain_vp(vp)

    for s in 1:S
        for p1 in 1:length(left_ids)
            p0 = left_ids[p1]
            vp_free[s][p0] = xs2[p1, s] ./ rescaling_free[p0]
        end

        # TODO: Simply treat kappa as a constrained variable.
        if ids.kappa[1, 1] in left_ids
            # Each column of kappa is the probability of being a 
            # type of celestial object (currently star or galax).
            # Here, assume that there are only two.
            vp_free[s][ids_free.kappa[end, :]] = 1. - vp_free[s][ids_free.kappa[1, :]]
        end
    end
    CelesteTypes.constrain_vp!(vp_free, vp)
end

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


function get_nlopt_bounds(vp::Vector{Vector{Float64}}, omitted_ids)
    lbs = [get_nlopt_bounds(vs)[1] for vs in vp]
    ubs = [get_nlopt_bounds(vs)[2] for vs in vp]
    vp_to_coordinates(lbs, omitted_ids), vp_to_coordinates(ubs, omitted_ids)
end


function get_nlopt_unconstrained_bounds(vp::Vector{Vector{Float64}}, omitted_ids)
    # Note that sources are not allowed to move more than
    # one pixel from their starting position in order to
    # avoid label switiching.  (This is why this function gets
    # the variational parameters as an argument.)
    #
    # vs: parameters for a particular source.
    # vp: complete collection of sources.

    lbs = [ get_nlopt_bounds(vs)[1] for vs in vp]
    ubs = [ get_nlopt_bounds(vs)[2] for vs in vp]
    vp_to_free_coordinates(lbs, omitted_ids), vp_to_free_coordinates(ubs, omitted_ids)
end


function print_params(vp)
    for vs in vp
        for n in names(ids)
            println(n, ": ", vs[ids.(n)])
        end
        println("-----------------\n")
    end
end


function maximize_f(f::Function, blob::Blob, mp::ModelParams; omitted_ids=Int64[],
    xtol_rel = 1e-7, ftol_abs = 1e-6)
    # This calls NLOpt
    x0 = vp_to_coordinates(mp.vp, omitted_ids)
    iter_count = 0

    function objective_and_grad(x::Vector{Float64}, g::Vector{Float64})
        println("Iter: ", iter_count)
        coordinates_to_vp!(x, mp.vp, omitted_ids)
        elbo = f(blob, mp)
        if length(g) > 0
            elbo2 = scale_deriv(elbo, omitted_ids, rescaling)
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
    lbs, ubs = get_nlopt_bounds(mp.vp, omitted_ids)
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


function maximize_unconstrained_f(f::Function, blob::Blob, mp::ModelParams; omitted_ids=Int64[],
    xtol_rel = 1e-7, ftol_abs = 1e-6)
    # Maximize using NLOpt and unconstrained coordinates.
    #
    # Args:
    #   - f: A function that takes a blob and constrianed coordinates (e.g. ElboDeriv.elbo)
    #   - blob: Input for f
    #   - mp: Constrained initial ModelParams
    #   - omitted_ids: Omitted ids from the _unconstrained_ parameterization (i.e. elements
    #       of free_ids).

    x0 = vp_to_free_coordinates(mp.vp, omitted_ids)
    iter_count = 0

    mp_free = deepcopy(mp)
    mp_free.vp = unconstrain_vp(mp_free.vp)

    function objective_and_grad(x::Vector{Float64}, g::Vector{Float64})
        println("Iter: ", iter_count)
        # Evaluate in the constrained space and then unconstrain again.
        free_coordinates_to_vp!(x, mp.vp, omitted_ids)
        elbo = f(blob, mp)
        elbo_free = ElboDeriv.unconstrain_sensitive_float(elbo, mp)
        if length(g) > 0
            elbo2 = scale_deriv(elbo_free, omitted_ids, rescaling_free)
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
    lbs, ubs = get_nlopt_unconstrained_bounds(mp_free.vp, omitted_ids)
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
