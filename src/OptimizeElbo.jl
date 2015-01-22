# written by Jeffrey Regier
# jeff [at] stat [dot] berkeley [dot] edu

module OptimizeElbo

using NLopt
using CelesteTypes

import ElboDeriv


const rescaling = ones(length(all_params))
rescaling[ids.chi] = 1e1
[rescaling[id] *= 1e1 for id in ids.mu]
[rescaling[id] *= 1e-5 for id in ids.gamma]
[rescaling[id] *= 1e1 for id in ids.zeta]
[rescaling[id] *= 1e2 for id in ids.kappa]
[rescaling[id] *= 1e1 for id in ids.theta]
[rescaling[id] *= 1e2 for id in ids.Xi]
[rescaling[id] *= 1e3 for id in ids.beta]


const omitted_ids = Int64[]
const left_ids = setdiff(all_params, omitted_ids)


function vs_to_coordinates(vs::Vector{Float64})
	ret = deepcopy(vs)
#=
	ret[ids.gamma] = vs[ids.gamma] .* vs[ids.zeta]  # mean
	ret[ids.zeta] = ret[ids.gamma] .* vs[ids.zeta]  # variance
=#
	ret .*= rescaling
	ret[left_ids]
end


function coordinates_to_vs(x::Vector{Float64})
	@assert length(x) == length(left_ids)
#=
	ret[ids.zeta] = x[ids.zeta] ./ ret[ids.gamma]
	ret[ids.gamma] = x[ids.gamma] ./ ret[ids.zeta]
=#
	ret = x ./ rescaling[left_ids]
	ret
end


function scale_deriv(x::Vector{Float64})
	ret = x ./ rescaling
	ret[left_ids]
end


function vp_to_coordinates(vp::Vector{Vector{Float64}})
	reduce(vcat, [vs_to_coordinates(vs) for vs in vp])
end


function coordinates_to_vp(xs::Vector{Float64})
	P = length(left_ids)
	@assert length(xs) % P == 0
	S = int(length(xs) / P)
	xs2 = reshape(xs, P, S)
	[coordinates_to_vs(xs2[:, s]) for s in 1:S]
end


function get_nlopt_bounds(vs::Vector{Float64})
	lb = Array(Float64, length(all_params))
	lb[ids.chi] = 1e-4
	lb[ids.mu] = vs[ids.mu] - 1.
	[lb[id] = 1e-4 for id in ids.gamma] #uggg...need min brightness
	[lb[id] = 1e-4 for id in ids.zeta]
	[lb[id] = 1e-4 for id in ids.kappa]
	[lb[id] = -1e4 for id in ids.beta]
	[lb[id] = 1e-4 for id in ids.lambda]
	lb[ids.theta] = 1e-2 
	[lb[id] = sqrt(2) for id in ids.Xi[[1,3]]]
	lb[ids.Xi[2]] = -10

	ub = Array(Float64, length(all_params))
	ub[ids.chi] = 1 - 1e-4
	ub[ids.mu] = vs[ids.mu] + 1.
	[ub[id] = 1e12 for id in ids.gamma]
	[ub[id] = 1e12 for id in ids.zeta]
	[ub[id] = 1 - 1e-4 for id in ids.kappa]
	ub[ids.theta] = 1 - 1e-2 
	[ub[id] = 10 for id in ids.Xi]
	[ub[id] = 1e4 for id in ids.beta]
	[ub[id] = 1e4 for id in ids.lambda]

	lb, ub
end


function get_nlopt_bounds(vp::Vector{Vector{Float64}})
	lbs = [get_nlopt_bounds(vs)[1] for vs in vp]
	ubs = [get_nlopt_bounds(vs)[2] for vs in vp]
	vp_to_coordinates(lbs), vp_to_coordinates(ubs)
end


function print_params(vp)
	for vs in vp
		for n in names(ids)
			println(n, ": ", vs[ids.(n)])
		end
		println("-----------------\n")
	end
end


function maximize_f(f::Function, blob::Blob, mp::ModelParams)
	x0 = vp_to_coordinates(mp.vp)
	iter_count = 0

	function objective_and_grad(x::Vector{Float64}, g::Vector{Float64})
		vp_new = coordinates_to_vp(x)
		for s in 1:mp.S
			mp.vp[s][left_ids] = vp_new[s]
		end
		elbo = f(blob, mp)
		if length(g) > 0
			svs = [scale_deriv(elbo.d[:, s]) for s in 1:mp.S]
			g[:] = reduce(vcat, svs)
		end

		iter_count += 1
		print_params(mp.vp)
		println("grad: ", g)
		println("elbo: ", elbo.v)
		println("\n=======================================\n")
		elbo.v
	end

	opt = Opt(:LD_LBFGS, length(x0))
	max_objective!(opt, objective_and_grad)
	xtol_rel!(opt, 1e-4)
	lbs, ubs = get_nlopt_bounds(mp.vp)
	lower_bounds!(opt, lbs)
	upper_bounds!(opt, ubs)
	(max_f, max_x, ret) = optimize(opt, x0)

	println("got $max_f at $max_x after $iter_count iterations (returned $ret)\n")
end


function maximize_elbo(blob::Blob, mp::ModelParams)
	empty!(omitted_ids)
	empty!(left_ids)
	append!(left_ids, all_params)
	maximize_f(ElboDeriv.elbo, blob, mp)
end


function maximize_likelihood(blob::Blob, mp::ModelParams)
	empty!(omitted_ids)
	append!(omitted_ids, [ids.kappa[:], ids.lambda[:], ids.zeta])
	empty!(left_ids)
	append!(left_ids, setdiff(all_params, omitted_ids))
	maximize_f(ElboDeriv.elbo_likelihood, blob, mp)
end

end
