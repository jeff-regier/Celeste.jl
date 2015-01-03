# written by Jeffrey Regier
# jeff [at] stat [dot] berkeley [dot] edu

module OptimizeElbo

using NLopt
using CelesteTypes

import ElboDeriv


const rescaling = ParamStruct{Float64}(
	1e1, (1e1, 1e1), (1e-6, 1e-6, 1e-6, 1e-6, 1e-6,), 1e-5,
	(1e-6, 1e-6, 1e-6, 1e-6, 1e-6,), 1e1, (1e0, 1e0, 1e0))

function rescale(x::Vector{Float64}, dir::Bool=true)
	@assert length(x) == 18
	z = dir ? 1. : -1.
	[x[p] * rescaling[p]^z for p in 1:18]
end

function pss_to_pvec(pss::Vector{ParamStruct{Float64}})
	reduce(vcat, [rescale(convert(Vector{Float64}, ps)) for ps in pss])
end

function pvec_to_pss(x::Vector{Float64})
	@assert length(x) % 18 == 0
	[convert(ParamStruct{Float64}, rescale(x[(18i-17):18i], false))
			for i=1:int(length(x)/18)]
end


function maximize_elbo(blob::Blob, mp::ModelParams)
	x0 = pss_to_pvec(mp.vp)

	function objective_and_grad(x::Vector{Float64}, g::Vector{Float64})
		mp.vp = pvec_to_pss(x)
		elbo = ElboDeriv.elbo(blob, mp)
		if length(g) > 0
			svs = [rescale(elbo.d[:, s], false) for s in 1:mp.S]
			g[:] = reduce(vcat, svs)
		end
		for vs in mp.vp
			println(show(vs))
			println("-----------------\n")
		end
		println("grad: ", g)
		println("elbo: ", elbo.v)
		println("\n=======================================\n")
		elbo.v
	end

	opt = Opt(:LD_LBFGS, length(x0))
	max_objective!(opt, objective_and_grad)
	xtol_rel!(opt, 1e-2)

	brightness_lb = (1e4, 1e4, 1e4, 1e4, 1e3)
    lb_s = ParamStruct{Float64}(1e-4, (-10., -10.), 
		brightness_lb, 1e3,
        brightness_lb, 1e-4, (sqrt(2), -10, sqrt(2)))
	lbs = pss_to_pvec([lb_s for i in 1:mp.S])
	lower_bounds!(opt, lbs)

	brightness_ub = (1e20, 1e20, 1e20, 1e20, 1e20) 
	H, W = blob[1].H, blob[1].W
    ub_s = ParamStruct{Float64}(1 - 1e-4, (H + 10, W + 10), 
		brightness_ub, 15_000,
        brightness_ub, 1. - 1e-4, (10., 10, 10))
	ubs = pss_to_pvec([ub_s for i in 1:mp.S])
	upper_bounds!(opt, ubs)

	(max_f, max_x, ret) = optimize(opt, x0)

	println("got $max_f at $max_x after $count iterations (returned $ret)\n")
    
	for vs in mp.vp
        print(show(vs))
        println("\n-----------------\n")
    end
end


end

