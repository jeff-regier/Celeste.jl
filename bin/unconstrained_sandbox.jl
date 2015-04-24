#!/usr/bin/env julia


# require(joinpath(Pkg.dir("Celeste"), "src", "Util.jl"))
# require(joinpath(Pkg.dir("Celeste"), "src", "CelesteTypes.jl"))
# require(joinpath(Pkg.dir("Celeste"), "src", "ElboDeriv.jl"))
# require(joinpath(Pkg.dir("Celeste"), "src", "OptimizeElbo.jl"))

#include(joinpath(Pkg.dir("Celeste"), "src", "Transform.jl"))
#include(joinpath(Pkg.dir("Celeste"), "src", "CelesteTypes.jl"))
#include(joinpath(Pkg.dir("Celeste"), "src", "ElboDeriv.jl"))
#include(joinpath(Pkg.dir("Celeste"), "src", "OptimizeElbo.jl"))


#include(joinpath(Pkg.dir("Celeste"), "test", "test_derivs.jl"))

using Celeste
using CelesteTypes
using Transform
using Base.Test

import OptimizeElbo
import ElboDeriv
import SampleData


blob, mp_init, body = SampleData.gen_sample_star_dataset();
#blob, mp_init, body = SampleData.gen_sample_galaxy_dataset();
#blob, mp_init, body = SampleData.gen_three_body_dataset();

mp_original = deepcopy(mp_init)
mp_free = deepcopy(mp_init)
mp_rect = deepcopy(mp_init)

# Optimize
omitted_ids = [ids_free.kappa[:], ids_free.lambda[:], ids_free.zeta[:] ]

lbs_free, ubs_free = OptimizeElbo.get_nlopt_unconstrained_bounds(mp_original.vp, omitted_ids, free_transform)
lbs_rect, ubs_rect= OptimizeElbo.get_nlopt_unconstrained_bounds(mp_original.vp, omitted_ids, rect_transform)

mp_free = deepcopy(mp_init)
res_free_iter_count, res_free_max_f, res_free_max_x, res_free_ret =
    OptimizeElbo.maximize_f(ElboDeriv.elbo_likelihood, blob, mp_free, free_transform,
    	       lbs_free, ubs_free, omitted_ids=omitted_ids, xtol_rel = 1e-7, ftol_abs = 1e-6)

mp_rect = deepcopy(mp_init)
res_rect_iter_count, res_rect_max_f, res_rect_max_x, res_rect_ret =
    OptimizeElbo.maximize_f(ElboDeriv.elbo_likelihood, blob, mp_rect, rect_transform,
      	                    lbs_rect, ubs_rect, omitted_ids=omitted_ids, xtol_rel = 1e-7, ftol_abs = 1e-6)



ElboDeriv.elbo_likelihood(blob, mp_rect).v
ElboDeriv.elbo_likelihood(blob, mp_free).v


# Compare the parameters, fits, and iterations.
println("===================")
println("Differences:")
for var_name in names(ids)
	println(var_name)
	for s in 1:mp_init.S
		println(s, ": ", mp_free.vp[s][ids.(var_name)] - mp_rect.vp[s][ids.(var_name)])
	end
end
println("===================")

res_free_max_f / res_rect_max_f
res_free_iter_count / res_rect_iter_count
