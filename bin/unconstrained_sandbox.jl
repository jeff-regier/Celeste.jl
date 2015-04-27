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

function compare_solutions(mp_rect::ModelParams, mp_free::ModelParams,
    res_rect_max_f, res_free_max_f, res_rect_iter_count, res_free_iter_count)
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

    println("Function value (free - rect) / abs(rect): ",
            (res_free_max_f - res_rect_max_f) / abs(res_rect_max_f))
    println("Iterations: (free / rect): ", res_free_iter_count / res_rect_iter_count)
end

########################################
# Three basic testing problems

#blob, mp_init, body = SampleData.gen_sample_star_dataset();
blob, mp_init, body = SampleData.gen_sample_galaxy_dataset();
#blob, mp_init, body = SampleData.gen_three_body_dataset();

mp_original = deepcopy(mp_init)
mp_free = deepcopy(mp_init)
mp_rect = deepcopy(mp_init)

# Optimize
omitted_ids = [ids_free.kappa[:], ids_free.lambda[:], ids_free.zeta[:] ]

lbs_free, ubs_free = OptimizeElbo.get_nlopt_unconstrained_bounds(mp_original.vp, omitted_ids, free_transform)
lbs_rect, ubs_rect= OptimizeElbo.get_nlopt_unconstrained_bounds(mp_original.vp, omitted_ids, rect_transform)

mp_rect = deepcopy(mp_init)
res_rect_iter_count, res_rect_max_f, res_rect_max_x, res_rect_ret =
    OptimizeElbo.maximize_f(ElboDeriv.elbo_likelihood, blob, mp_rect, rect_transform,
                            lbs_rect, ubs_rect, omitted_ids=omitted_ids, xtol_rel = 1e-7, ftol_abs = 1e-6)

mp_free = deepcopy(mp_init)
res_free_iter_count, res_free_max_f, res_free_max_x, res_free_ret =
    OptimizeElbo.maximize_f(ElboDeriv.elbo_likelihood, blob, mp_free, free_transform,
    	       lbs_free, ubs_free, omitted_ids=omitted_ids, xtol_rel = 1e-7, ftol_abs = 1e-6)

compare_solutions(mp_rect, mp_free, res_rect_max_f, res_free_max_f, res_rect_iter_count, res_free_iter_count)



########################################
# Three body

blob, mp_init, body = SampleData.gen_three_body_dataset();

mp_original = deepcopy(mp_init)
mp_free = deepcopy(mp_init)
mp_rect = deepcopy(mp_init)

# Optimize
omitted_ids = [ids_free.kappa[:], ids_free.lambda[:], ids_free.zeta[:] ]

lbs_free, ubs_free = OptimizeElbo.get_nlopt_unconstrained_bounds(mp_original.vp, omitted_ids, free_transform)
lbs_rect, ubs_rect= OptimizeElbo.get_nlopt_unconstrained_bounds(mp_original.vp, omitted_ids, rect_transform)

mp_rect_three = deepcopy(mp_init)
res_rect_iter_count_three, res_rect_max_f_three, res_rect_max_x, res_rect_ret =
    OptimizeElbo.maximize_f(ElboDeriv.elbo_likelihood, blob, mp_rect_three, rect_transform,
                            lbs_rect, ubs_rect, omitted_ids=omitted_ids, xtol_rel = 1e-7, ftol_abs = 1e-6)

mp_free_three = deepcopy(mp_init)
res_free_iter_count_three, res_free_max_f_three, res_free_max_x, res_free_ret =
    OptimizeElbo.maximize_f(ElboDeriv.elbo_likelihood, blob, mp_free_three, free_transform,
               lbs_free, ubs_free, omitted_ids=omitted_ids, xtol_rel = 1e-7, ftol_abs = 1e-6)

compare_solutions(mp_rect_three, mp_free_three,
    res_rect_max_f_three, res_free_max_f_three,
    res_rect_iter_count_three, res_free_iter_count_three)



########################################
# From test_peak_init_2body_optimization
srand(1)
blob0 = SDSS.load_stamp_blob(SampleData.dat_dir, "164.4311-39.0359");
two_bodies = [
    SampleData.sample_ce([11.1, 21.2], true),
    SampleData.sample_ce([15.3, 31.4], false),
];

blob = Synthetic.gen_blob(blob0, two_bodies);
mp = ModelInit.peak_init(blob); #one giant tile, giant patches

mp_rect_2body = deepcopy(mp)
res_rect_iter_count_2body, res_rect_max_f_2body, res_rect_max_x, res_rect_ret =
    OptimizeElbo.maximize_likelihood(blob, mp_rect_2body, rect_transform)

mp_free_2body = deepcopy(mp)
res_free_iter_count_2body, res_free_max_f_2body, res_free_max_x, res_free_ret =
    OptimizeElbo.maximize_likelihood(blob, mp_free_2body, free_transform)

compare_solutions(mp_rect_2body, mp_free_2body,
    res_rect_max_f_2body, res_free_max_f_2body,
    res_rect_iter_count_2body, res_free_iter_count_2body)


#################################
# From test_full_elbo_optimization

blob, mp, body = SampleData.gen_sample_galaxy_dataset(perturb=true);

mp_rect_elbo = deepcopy(mp);
res_rect_iter_count_elbo, res_rect_max_f_elbo, res_rect_max_x, res_rect_ret =
    OptimizeElbo.maximize_elbo(blob, mp_rect_elbo, rect_transform)

mp_free_elbo = deepcopy(mp);
res_free_iter_count_elbo, res_free_max_f_elbo, res_free_max_x, res_free_ret =
    OptimizeElbo.maximize_elbo(blob, mp_free_elbo, free_transform)

compare_solutions(mp_rect_elbo, mp_free_elbo,
    res_rect_max_f_elbo, res_free_max_f_elbo,
    res_rect_iter_count_elbo, res_free_iter_count_elbo)




#############################################
# From test_real_stamp_optimization.  This is slow.
blob = SDSS.load_stamp_blob(SampleData.dat_dir, "5.0073-0.0739");
cat_entries = SDSS.load_stamp_catalog(SampleData.dat_dir, "s82-5.0073-0.0739", blob);
bright(ce) = sum(ce.star_fluxes) > 3 || sum(ce.gal_fluxes) > 3;
cat_entries = filter(bright, cat_entries);
inbounds(ce) = ce.pos[1] > -10. && ce.pos[2] > -10 &&
    ce.pos[1] < 61 && ce.pos[2] < 61;
cat_entries = filter(inbounds, cat_entries);

mp = ModelInit.cat_init(cat_entries);

mp_rect_real = deepcopy(mp);
res_rect_iter_count_real, res_rect_max_f_real, res_rect_max_x, res_rect_ret =
    OptimizeElbo.maximize_elbo(blob, mp_rect_real, rect_transform)

mp_free_real = deepcopy(mp);
res_free_iter_count_real, res_free_max_f_real, res_free_max_x, res_free_ret =
    OptimizeElbo.maximize_elbo(blob, mp_free_real, free_transform)




