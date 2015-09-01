#!/usr/bin/env julia

using Celeste
using CelesteTypes

import Synthetic
import ModelInit
import SampleData

const dat_dir = joinpath(Pkg.dir("Celeste"), "dat")

function small_image_profile()
    srand(1)
    blob, mp, body, tiled_blob =
      SampleData.gen_n_body_dataset(100, tile_width=10);
    println("Calculating ELBO.")
    elbo = ElboDeriv.elbo(tiled_blob, mp)
end


println("Running with ", length(workers()), " processors.")

Profile.init(10^7, 0.001)
small_image_profile()
@profile small_image_profile()
#Profile.print()
Profile.print(format=:flat)
