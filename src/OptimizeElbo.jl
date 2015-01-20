# written by Jeffrey Regier
# jeff [at] stat [dot] berkeley [dot] edu

module OptimizeElbo

using NLopt
using CelesteTypes

import ElboDeriv


const rescaling = ones(length(all_params))
rescaling[ids.chi] = 1e1
[rescaling[id] *= 1e1 for id in ids.mu]
[rescaling[id] *= 1e1 for id in ids.gamma]
[rescaling[id] *= 1e-9 for id in ids.zeta]
[rescaling[id] *= 1e2 for id in ids.kappa]
[rescaling[id] *= 1e1 for id in ids.theta]
[rescaling[id] *= 1e1 for id in ids.Xi]


function rescale_s(x::Vector{Float64}, dir::Bool=true)
	@assert length(x) == length(all_params)
	ret = deepcopy(x)
#=
	if dir
		ret[ids.gamma] = x[ids.gamma] .* x[ids.zeta]  # mean
		ret[ids.zeta] = ret[ids.gamma] .* x[ids.zeta]  # variance
	end

	z = dir ? 1. : -1.
	ret .*= rescaling.^z

	if !dir
		ret[ids.zeta] = x[ids.zeta] ./ ret[ids.gamma]
		ret[ids.gamma] = x[ids.gamma] ./ ret[ids.zeta]
	end
=#
	ret
end

function vp_to_vec(vp::Vector{Vector{Float64}})
	ret = Array(Float64, length(all_params), length(vp))
	for s in 1:length(vp)
		ret[:, s] = rescale_s(vp[s], true)
	end
	ret[:]
end

function vec_to_vp(pvec::Vector{Float64})
	@assert length(pvec) % length(all_params) == 0
	S = int(length(pvec) / length(all_params))
	vec2 = reshape(pvec, length(all_params), S)
	ret = Array(Vector{Float64}, S)
	for s in 1:S
		ret[s] = rescale_s(vec2[: ,s], false)
	end
	ret
end


function get_nlopt_bounds(img, S)
	lb = Array(Float64, length(all_params))
	lb[ids.chi] = 1e-4
	[lb[id] = -10. for id in ids.mu]
	[lb[id] = 1e-4 for id in ids.gamma] #uggg...need min brightness
	[lb[id] = 1e-4 for id in ids.zeta]
	[lb[id] = 1e-4 for id in ids.kappa]
	[lb[id] = -1e4 for id in ids.beta]
	[lb[id] = 1e-4 for id in ids.lambda]
	lb[ids.theta] = 1e-2 
	[lb[id] = sqrt(2) for id in ids.Xi[[1,3]]]
	lb[ids.Xi[2]] = -10
	lb2 = rescale_s(lb)
	lbs = reduce(vcat, [deepcopy(lb2) for s in 1:S])

	ub = Array(Float64, length(all_params))
	ub[ids.chi] = 1 - 1e-4
	ub[ids.mu] = [img.H + 10, img.W + 10]
	[ub[id] = 1e20 for id in ids.gamma]
	[ub[id] = 1e20 for id in ids.zeta]
	[ub[id] = 1 - 1e-4 for id in ids.kappa]
	ub[ids.theta] = 1 - 1e-2 
	[ub[id] = 10 for id in ids.Xi]
	[ub[id] = 1e4 for id in ids.beta]
	[ub[id] = 1e4 for id in ids.lambda]
	ub2 = rescale_s(ub)
	ubs = reduce(vcat, [deepcopy(ub2) for s in 1:S])

	lbs, ubs
end


function print_params(vp)
	for vs in vp
		for n in names(ids)
			println(n, ": ", vs[ids.(n)])
		end
		println("-----------------\n")
	end
end


function maximize_elbo(blob::Blob, mp::ModelParams)
	x0 = vp_to_vec(mp.vp)

	function objective_and_grad(x::Vector{Float64}, g::Vector{Float64})
		mp.vp = vec_to_vp(x)
#		elbo = ElboDeriv.elbo(blob, mp)
		elbo = zero_sensitive_float([1:mp.S], all_params)
		elbo.v = sum([sum(vs) for vs in mp.vp])
		fill!(elbo.d, 1.)
		if length(g) > 0
			svs = [rescale_s(elbo.d[:, s], false) for s in 1:mp.S]
			g[:] = reduce(vcat, svs)
		end

		print_params(mp.vp)
		println("grad: ", g)
		println("elbo: ", elbo.v)
		println("\n=======================================\n")
		elbo.v
	end

	opt = Opt(:LD_LBFGS, length(x0))
	max_objective!(opt, objective_and_grad)
	xtol_rel!(opt, 1e-4)
	lbs, ubs = get_nlopt_bounds(blob[1], mp.S)
	lower_bounds!(opt, lbs)
	upper_bounds!(opt, ubs)
	(max_f, max_x, ret) = optimize(opt, x0)

	println("got $max_f at $max_x after $count iterations (returned $ret)\n")
end


end

