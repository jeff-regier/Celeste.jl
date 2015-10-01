#!/usr/bin/env julia

using Celeste
using CelesteTypes

import Synthetic
import ModelInit
import SampleData

const dat_dir = joinpath(Pkg.dir("Celeste"), "dat")

srand(1)
println("Loading data.")

S = 100
blob, mp, body, tiled_blob =
  SampleData.gen_n_body_dataset(S, tile_width=10);

function small_image_profile()
    println("Calculating ELBO.")
    elbo_time = time()
    elbo = ElboDeriv.elbo(tiled_blob, mp);
    elbo_time = time() - elbo_time
    elbo, elbo_time
end

println("Running with ", length(workers()), " processors.")
@time elbo, elbo_time = small_image_profile();

Profile.init(10^7, 0.001)
@profile small_image_profile()
Profile.print()

println("Elbo time: $(elbo_time) seconds")
#Profile.print(format=:flat)
