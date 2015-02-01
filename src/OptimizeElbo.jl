# written by Jeffrey Regier
# jeff [at] stat [dot] berkeley [dot] edu

module OptimizeElbo

using NLopt
using CelesteTypes

import ElboDeriv


const xtol_rel = 1e-7
const ftol_abs = 1e-6

const rescaling = ones(length(all_params))
[rescaling[id] *= 1e-3 for id in ids.gamma]
rescaling[ids.chi] *= 1e1


function scale_deriv(elbo::SensitiveFloat, omitted_ids)
	left_ids = setdiff(all_params, [omitted_ids, ids.kappa[end, :][:]])
	new_P = length(left_ids)

	elbo_new = zero_sensitive_float(elbo.source_index, left_ids)
	elbo_new.v = elbo.v	

	for p1 in 1:length(left_ids)
		p0 = left_ids[p1]
		elbo_new.d[p1, :] = elbo.d[p0, :] ./ rescaling[p0]

		for i = 1:2
			if p0 == ids.kappa[1, i]
				elbo_new.d[p1, :] -= elbo.d[ids.kappa[end, i]]
			end
		end
	end
	
	elbo_new
end


function vp_to_coordinates(vp::Vector{Vector{Float64}}, omitted_ids::Vector{Int64})
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
			vp[s][ids.kappa[end, :]] = 1. - vp[s][ids.kappa[1, :]]
		end
	end
end


function get_nlopt_bounds(vs::Vector{Float64})
	lb = Array(Float64, length(all_params))
	lb[ids.chi] = 1e-4
	lb[ids.mu] = vs[ids.mu] - 1.
	[lb[id] = 1e-4 for id in ids.gamma]
	[lb[id] = 1e-4 for id in ids.zeta]
	[lb[id] = 1e-4 for id in ids.kappa]
	[lb[id] = -10. for id in ids.beta]
	[lb[id] = 1e-4 for id in ids.lambda]
	lb[ids.theta] = 1e-4 
	lb[ids.rho] = 1e-4
	lb[ids.phi] = -1e10 #-pi/2 + 1e-4
	lb[ids.sigma] = 0.2

	ub = Array(Float64, length(all_params))
	ub[ids.chi] = 1 - 1e-4
	ub[ids.mu] = vs[ids.mu] + 1.
	[ub[id] = 1e12 for id in ids.gamma]
	[ub[id] = 1e-1 for id in ids.zeta]
	[ub[id] = 1 - 1e-4 for id in ids.kappa]
	ub[ids.theta] = 1 - 1e-4
	ub[ids.rho] = 1. - 1e-4
	ub[ids.phi] = 1e10 #pi/2 - 1e-4
	ub[ids.sigma] = 50.
	[ub[id] = 10. for id in ids.beta]
	[ub[id] = 1. for id in ids.lambda]

	lb, ub
end


function get_nlopt_bounds(vp::Vector{Vector{Float64}}, omitted_ids)
	lbs = [get_nlopt_bounds(vs)[1] for vs in vp]
	ubs = [get_nlopt_bounds(vs)[2] for vs in vp]
	vp_to_coordinates(lbs, omitted_ids), vp_to_coordinates(ubs, omitted_ids)
end


function print_params(vp)
	for vs in vp
		for n in names(ids)
			println(n, ": ", vs[ids.(n)])
		end
		println("-----------------\n")
	end
end


function maximize_f(f::Function, blob::Blob, mp::ModelParams; omitted_ids=Int64[])
	x0 = vp_to_coordinates(mp.vp, omitted_ids)
	iter_count = 0

	function objective_and_grad(x::Vector{Float64}, g::Vector{Float64})
		coordinates_to_vp!(x, mp.vp, omitted_ids)
		elbo = f(blob, mp)
		if length(g) > 0
			elbo2 = scale_deriv(elbo, omitted_ids)
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
end


function maximize_elbo(blob::Blob, mp::ModelParams)
#	maximize_likelihood(blob, mp)
	maximize_f(ElboDeriv.elbo, blob, mp)
end


function maximize_likelihood(blob::Blob, mp::ModelParams)
	omitted_ids = [ids.kappa[:], ids.lambda[:], ids.zeta]
	maximize_f(ElboDeriv.elbo_likelihood, blob, mp, omitted_ids=omitted_ids)
end

end
