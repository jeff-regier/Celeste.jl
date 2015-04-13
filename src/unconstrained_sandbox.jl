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


const sample_star_fluxes = [
    4.451805E+03,1.491065E+03,2.264545E+03,2.027004E+03,1.846822E+04]
const sample_galaxy_fluxes = [
    1.377666E+01, 5.635334E+01, 1.258656E+02, 
    1.884264E+02, 2.351820E+02] * 100  # 1x wasn't bright enough


function sample_ce(pos, is_star::Bool)
    CatalogEntry(pos, is_star, sample_star_fluxes, sample_galaxy_fluxes, 
        0.1, .7, pi/4, 4.)
end

blob0 = SDSS.load_stamp_blob(dat_dir, "164.4311-39.0359")
for b in 1:5
    blob0[b].H, blob0[b].W = 112, 238
end

three_bodies = [
    sample_ce([4.5, 3.6], false),
    sample_ce([60.1, 82.2], true),
    sample_ce([71.3, 100.4], false),
];

blob = Synthetic.gen_blob(blob0, three_bodies);
vp = [ModelInit.init_source(ce) for ce in three_bodies]
patches = [SkyPatch(ce.pos, 20) for ce in three_bodies]
my_prior = ModelInit.sample_prior()
mp = ModelParams(vp, my_prior, patches, 1000)


mp = ModelInit.cat_init(three_bodies, patch_radius=20., tile_width=1000)
ret = ElboDeriv.elbo(blob, mp)

ret_free = ElboDeriv.unconstrain_sensitive_float(ret, mp)

omitted_indices = Array(Int64, 0)
x_free = OptimizeElbo.vp_to_free_coordinates(vp, omitted_indices)
x = OptimizeElbo.vp_to_coordinates(vp, omitted_indices)
hcat(x_free, x)

vp_new1 = copy(vp)
vp_new2 = copy(vp)
OptimizeElbo.coordinates_to_vp!(x, vp_new1, omitted_indices)
OptimizeElbo.free_coordinates_to_vp!(x_free, vp_new2, omitted_indices)





