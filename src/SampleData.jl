module SampleData

using Distributions
using CelesteTypes

import SDSS
import ModelInit
import Synthetic


export empty_model_params
export dat_dir, sample_ce, perturb_params
export sample_star_fluxes, sample_galaxy_fluxes
export gen_sample_star_dataset, gen_sample_galaxy_dataset, gen_three_body_dataset

const dat_dir = joinpath(Pkg.dir("Celeste"), "dat")

const sample_star_fluxes = [
    4.451805E+03,1.491065E+03,2.264545E+03,2.027004E+03,1.846822E+04]
const sample_galaxy_fluxes = [
    1.377666E+01, 5.635334E+01, 1.258656E+02,
    1.884264E+02, 2.351820E+02] * 100  # 1x wasn't bright enough


function empty_model_params(S::Int)
    vp = [ ModelInit.init_source([ 0., 0. ]) for s in 1:S ]
    patches = [ SkyPatch([ 0., 0., ], 1.) for s in 1:S ]
    ModelParams(vp, ModelInit.sample_prior(), patches, 1.)
end


function sample_ce(pos, is_star::Bool)
    CatalogEntry(pos, is_star, sample_star_fluxes, sample_galaxy_fluxes,
        0.1, .7, pi/4, 4.)
end


function perturb_params(mp) # for testing derivatives != 0
    for vs in mp.vp
        vs[ids.a] = [ 0.4, 0.6 ]
        vs[ids.u[1]] += .8
        vs[ids.u[2]] -= .7
        vs[ids.r1] /= 10
        vs[ids.r2] *= 25.
        vs[ids.e_dev] += 0.05
        vs[ids.e_axis] += 0.05
        vs[ids.e_angle] += pi/10
        vs[ids.e_scale] *= 1.2
        vs[ids.c1] += 0.5
        vs[ids.c2] =  1e-1
    end
end


function gen_sample_star_dataset(; perturb=true)
    srand(1)
    blob0 = SDSS.load_stamp_blob(dat_dir, "164.4311-39.0359")
    for b in 1:5
        blob0[b].H, blob0[b].W = 20, 23
    end
    one_body = [sample_ce([10.1, 12.2], true),]
       blob = Synthetic.gen_blob(blob0, one_body)
    mp = ModelInit.cat_init(one_body)
    if perturb
        perturb_params(mp)
    end

    blob, mp, one_body
end


function gen_sample_galaxy_dataset(; perturb=true)
    srand(1)
    blob0 = SDSS.load_stamp_blob(dat_dir, "164.4311-39.0359")
    for b in 1:5
        blob0[b].H, blob0[b].W = 20, 23
    end
    one_body = [sample_ce([8.5, 9.6], false),]
    blob = Synthetic.gen_blob(blob0, one_body)
    mp = ModelInit.cat_init(one_body)
    if perturb
        perturb_params(mp)
    end

    blob, mp, one_body
end


function gen_three_body_dataset(; perturb=true)
    srand(1)
    blob0 = SDSS.load_stamp_blob(dat_dir, "164.4311-39.0359")
    for b in 1:5
        blob0[b].H, blob0[b].W = 112, 238
    end
    three_bodies = [
        sample_ce([4.5, 3.6], false),
        sample_ce([60.1, 82.2], true),
        sample_ce([71.3, 100.4], false),
    ]
    blob = Synthetic.gen_blob(blob0, three_bodies)
    mp = ModelInit.cat_init(three_bodies)
    if perturb
        perturb_params(mp)
    end

    blob, mp, three_bodies
end

end # End module
