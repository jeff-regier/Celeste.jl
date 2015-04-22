#!/usr/bin/env julia


require(joinpath(Pkg.dir("Celeste"), "src", "Util.jl"))
require(joinpath(Pkg.dir("Celeste"), "src", "CelesteTypes.jl"))
#require(joinpath(Pkg.dir("Celeste"), "src", "ModelInit.jl"))
#require(joinpath(Pkg.dir("Celeste"), "src", "SDSS.jl"))
require(joinpath(Pkg.dir("Celeste"), "src", "ElboDeriv.jl"))
require(joinpath(Pkg.dir("Celeste"), "src", "OptimizeElbo.jl"))
#require(joinpath(Pkg.dir("Celeste"), "src", "Synthetic.jl"))

#include(joinpath(Pkg.dir("Celeste"), "src", "Constrain.jl"))
#include(joinpath(Pkg.dir("Celeste"), "src", "CelesteTypes.jl"))
#include(joinpath(Pkg.dir("Celeste"), "src", "ElboDeriv.jl"))
#include(joinpath(Pkg.dir("Celeste"), "src", "OptimizeElbo.jl"))

using Celeste

import ElboDeriv
using CelesteTypes
import OptimizeElbo

using Base.Test
using SampleData

blob, mp_init, body = gen_sample_star_dataset();
#blob, mp_init, body = gen_sample_galaxy_dataset();
#blob, mp_init, body = gen_three_body_dataset();

mp_original = deepcopy(mp_init)
mp_free = deepcopy(mp_init)

# Optimize
omitted_ids = [ids.kappa[:], ids.lambda[:], ids.zeta[:] ]
omitted_ids_free = [ids_free.kappa[:], ids_free.lambda[:], ids_free.zeta[:] ]

res_free_iter_count, res_free_max_f, res_free_max_x, res_free_ret =
    OptimizeElbo.maximize_unconstrained_likelihood(blob, mp_free)



# This should not work anymore:
res_iter_count, res_max_f, res_max_x, res_ret =
	OptimizeElbo.maximize_likelihood(blob, mp_original)

mp = deepcopy(mp_free)
x0 = OptimizeElbo.vp_to_free_coordinates(mp.vp, omitted_ids, unconstrain_vp)
OptimizeElbo.free_coordinates_to_vp!(x0, mp.vp, omitted_ids,
                        unconstrain_vp, constrain_vp!)

OptimizeElbo.free_coordinates_to_vp!(x0, mp.vp, omitted_ids,
                        unconstrain_vp, constrain_vp!)
elbo = ElboDeriv.elbo_likelihood(blob, mp)
elbo_free = ElboDeriv.unconstrain_sensitive_float(elbo, mp)
OptimizeElbo.print_params(mp.vp)




# Compare the parameters, fits, and iterations.
println("===================")
println("Differences:")
for var_name in names(ids)
	println(var_name)
	for s in 1:mp_init.S
		println(s, ": ", mp_free.vp[s][ids.(var_name)] - mp_original.vp[s][ids.(var_name)])
	end
end
println("===================")

res_free_max_f / res_max_f
res_free_iter_count / res_iter_count





# const dat_dir = joinpath(Pkg.dir("Celeste"), "dat")

# function sample_ce(pos, is_star::Bool)
#     CatalogEntry(pos, is_star, sample_star_fluxes, sample_galaxy_fluxes, 
#         0.1, .7, pi/4, 4.)
# end
# const sample_star_fluxes = [
#     4.451805E+03,1.491065E+03,2.264545E+03,2.027004E+03,1.846822E+04]
# const sample_galaxy_fluxes = [
#     1.377666E+01, 5.635334E+01, 1.258656E+02, 
#     1.884264E+02, 2.351820E+02] * 100  # 1x wasn't bright enough


# function perturb_params(mp) # for testing derivatives != 0
#     for vs in mp.vp
#         vs[ids.chi[2]] = 0.6
#         vs[ids.chi[1]] = 1.0 - vs[ids.chi[2]]
#         vs[ids.mu[1]] += .8
#         vs[ids.mu[2]] -= .7
#         vs[ids.gamma] /= 10
#         vs[ids.zeta] *= 25.
#         vs[ids.theta] += 0.05
#         vs[ids.rho] += 0.05
#         vs[ids.phi] += pi/10
#         vs[ids.sigma] *= 1.2
#         vs[ids.beta] += 0.5
#         vs[ids.lambda] =  1e-1
#     end
# end

# function gen_sample_star_dataset(; perturb=true)
#     srand(1)
#     blob0 = SDSS.load_stamp_blob(dat_dir, "164.4311-39.0359")
#     for b in 1:5
#         blob0[b].H, blob0[b].W = 20, 23
#     end
#     one_body = [sample_ce([10.1, 12.2], true),]
#        blob = Synthetic.gen_blob(blob0, one_body)
#     mp = ModelInit.cat_init(one_body)
#     if perturb
#         perturb_params(mp)
#     end

#     blob, mp, one_body
# end


# function gen_sample_galaxy_dataset(; perturb=true)
#     srand(1)
#     blob0 = SDSS.load_stamp_blob(dat_dir, "164.4311-39.0359")
#     for b in 1:5
#         blob0[b].H, blob0[b].W = 20, 23
#     end
#     one_body = [sample_ce([8.5, 9.6], false),]
#     blob = Synthetic.gen_blob(blob0, one_body)
#     mp = ModelInit.cat_init(one_body)
#     if perturb
#         perturb_params(mp)
#     end

#     blob, mp, one_body
# end


# function gen_three_body_dataset(; perturb=true)
#     srand(1)
#     blob0 = SDSS.load_stamp_blob(dat_dir, "164.4311-39.0359")
#     for b in 1:5
#         blob0[b].H, blob0[b].W = 112, 238
#     end
#     three_bodies = [
#         sample_ce([4.5, 3.6], false),
#         sample_ce([60.1, 82.2], true),
#         sample_ce([71.3, 100.4], false),
#     ]
#     blob = Synthetic.gen_blob(blob0, three_bodies)
#     mp = ModelInit.cat_init(three_bodies)
#     if perturb
#         perturb_params(mp)
#     end

#     blob, mp, three_bodies
# end

