#!/usr/bin/env julia


require(joinpath(Pkg.dir("Celeste"), "src", "Util.jl"))
require(joinpath(Pkg.dir("Celeste"), "src", "CelesteTypes.jl"))
require(joinpath(Pkg.dir("Celeste"), "src", "ModelInit.jl"))
require(joinpath(Pkg.dir("Celeste"), "src", "SDSS.jl"))
require(joinpath(Pkg.dir("Celeste"), "src", "ElboDeriv.jl"))
require(joinpath(Pkg.dir("Celeste"), "src", "OptimizeElbo.jl"))
require(joinpath(Pkg.dir("Celeste"), "src", "Synthetic.jl"))

#include(joinpath(Pkg.dir("Celeste"), "src", "CelesteTypes.jl"))
#include(joinpath(Pkg.dir("Celeste"), "src", "ElboDeriv.jl"))
#include(joinpath(Pkg.dir("Celeste"), "src", "OptimizeElbo.jl"))

import ElboDeriv
using CelesteTypes

import SDSS
import OptimizeElbo
import ModelInit

using Celeste
using Base.Test

using Distributions

import Synthetic

const dat_dir = joinpath(Pkg.dir("Celeste"), "dat")

function sample_ce(pos, is_star::Bool)
    CatalogEntry(pos, is_star, sample_star_fluxes, sample_galaxy_fluxes, 
        0.1, .7, pi/4, 4.)
end
const sample_star_fluxes = [
    4.451805E+03,1.491065E+03,2.264545E+03,2.027004E+03,1.846822E+04]
const sample_galaxy_fluxes = [
    1.377666E+01, 5.635334E+01, 1.258656E+02, 
    1.884264E+02, 2.351820E+02] * 100  # 1x wasn't bright enough


function perturb_params(mp) # for testing derivatives != 0
    for vs in mp.vp
        vs[ids.chi] = 0.6
        vs[ids.mu[1]] += .8
        vs[ids.mu[2]] -= .7
        vs[ids.gamma] /= 10
        vs[ids.zeta] *= 25.
        vs[ids.theta] += 0.05
        vs[ids.rho] += 0.05
        vs[ids.phi] += pi/10
        vs[ids.sigma] *= 1.2
        vs[ids.beta] += 0.5
        vs[ids.lambda] =  1e-1
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


blob, mp, body = gen_sample_star_dataset();

vp = mp.vp
vp_free = unconstrain_vp(vp)

ret = ElboDeriv.elbo(blob, mp)
ret_free = ElboDeriv.unconstrain_sensitive_float(ret, mp)

omitted_indices = Array(Int64, 0)
x_free = OptimizeElbo.vp_to_free_coordinates(vp, omitted_indices)
x = OptimizeElbo.vp_to_coordinates(vp, omitted_indices)
# Should differ in the chi place.
hcat(x_free, x)

# Should be the same
vp_new1 = deepcopy(vp)
vp_new2 = deepcopy(vp)
OptimizeElbo.coordinates_to_vp!(x, vp_new1, omitted_indices)
OptimizeElbo.free_coordinates_to_vp!(x_free, vp_new2, omitted_indices)
@assert vp_new1 == vp_new2

# Should change
x_free2 = deepcopy(x_free)
x_free2[1] += 0.2
@assert x_free2[1] == x_free[1] + 0.2
OptimizeElbo.free_coordinates_to_vp!(x_free2, vp_new2, omitted_indices)
@assert vp_new1[1][1] != vp_new2[1][1]
vp_new1[1][1], vp_new2[1][1]



# Optimize
omitted_ids = [ids_free.kappa[:], ids_free.lambda[:], ids_free.zeta]
res = OptimizeElbo.maximize_f(ElboDeriv.elbo_likelihood, blob, mp, omitted_ids=omitted_ids)

mp2 = deepcopy(mp)
res_free = OptimizeElbo.maximize_unconstrained_f(ElboDeriv.elbo_likelihood, blob,
												 mp2, omitted_ids=omitted_ids)


