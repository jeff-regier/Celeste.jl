#!/usr/bin/env julia

using Celeste
using CelesteTypes

import Synthetic


const dat_dir = joinpath(Pkg.dir("Celeste"), "dat")


function small_image_profile()
    srand(1)

    blob0 = SDSS.load_stamp_blob(dat_dir, "164.4311-39.0359")
    for b in 1:5
        blob0[b].H, blob0[b].W = 1000, 1000
    end

    fluxes = [4.451805E+03,1.491065E+03,2.264545E+03,2.027004E+03,1.846822E+04]

    S = 100
    locations = rand(2, S) .* 1000.

    S_bodies = CatalogEntry[CatalogEntry(locations[:, s][:], true, 
        fluxes, fluxes, 0.1, .7, pi/4, 4.) for s in 1:S]

    blob = Synthetic.gen_blob(blob0, S_bodies)
    mp = ModelInit.cat_init(S_bodies, patch_radius=20., tile_width=10)
    elbo = ElboDeriv.elbo(blob, mp)
end


Profile.init(10^7, 0.001)
small_image_profile()
@profile small_image_profile()
#Profile.print()
Profile.print(format=:flat)


