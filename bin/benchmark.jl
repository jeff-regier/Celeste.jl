#!/usr/bin/env julia

using Celeste
using CelesteTypes

import Synthetic
import ModelInit
import SampleData

const dat_dir = joinpath(Pkg.dir("Celeste"), "dat")

srand(1)
println("Loading data.")
blob, mp, body, tiled_blob =
  SampleData.gen_n_body_dataset(100, tile_width=10);

function small_image_profile()
    println("Calculating ELBO.")
    elbo_time = time()
    elbo = ElboDeriv.elbo(tiled_blob, mp)
    elbo_time = time() - elbo_time
    elbo, elbo_time
end


println("Running with ", length(workers()), " processors.")

Profile.init(10^7, 0.001)
elbo, elbo_time = small_image_profile()
@profile small_image_profile()
Profile.print()
#Profile.print(format=:flat)
