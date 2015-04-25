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

# blob, mp, body = gen_sample_star_dataset();
# OptimizeElbo.maximize_likelihood(blob, mp, free_transform, xtol_rel=1e-16, ftol_abs=1e-10)
# #-529466.9212112223, 248
# verify_sample_star(mp.vp[1], [10.1, 12.2])

# blob, mp, body = gen_sample_star_dataset();
# OptimizeElbo.maximize_likelihood(blob, mp, rect_transform, xtol_rel=1e-16, ftol_abs=1e-10)
# # -529466.9212112254, 262
# verify_sample_star(mp.vp[1], [10.1, 12.2])



# From test_peak_init_2body_optimization
srand(1)
blob0 = SDSS.load_stamp_blob(dat_dir, "164.4311-39.0359")

two_bodies = [
    sample_ce([11.1, 21.2], true),
    sample_ce([15.3, 31.4], false),
]

blob = Synthetic.gen_blob(blob0, two_bodies)
mp = ModelInit.peak_init(blob) #one giant tile, giant patches
@test mp.S == 2

OptimizeElbo.maximize_likelihood(blob, mp, trans)

verify_sample_star(mp.vp[1], [11.1, 21.2])
verify_sample_galaxy(mp.vp[2], [15.3, 31.4])


# From test_full_elbo_optimization
blob, mp, body = gen_sample_galaxy_dataset(perturb=true)
OptimizeElbo.maximize_elbo(blob, mp, trans)
verify_sample_galaxy(mp.vp[1], [8.5, 9.6])



# From test_real_stamp_optimization.  This is slow.
blob = SDSS.load_stamp_blob(dat_dir, "5.0073-0.0739")
cat_entries = SDSS.load_stamp_catalog(dat_dir, "s82-5.0073-0.0739", blob)
bright(ce) = sum(ce.star_fluxes) > 3 || sum(ce.gal_fluxes) > 3
cat_entries = filter(bright, cat_entries)
inbounds(ce) = ce.pos[1] > -10. && ce.pos[2] > -10 &&
    ce.pos[1] < 61 && ce.pos[2] < 61
cat_entries = filter(inbounds, cat_entries)

mp = ModelInit.cat_init(cat_entries)
OptimizeElbo.maximize_elbo(blob, mp, trans)



