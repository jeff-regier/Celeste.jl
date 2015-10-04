#!/usr/bin/env julia

using Celeste
using CelesteTypes

import Synthetic
import ModelInit
import SampleData

const dat_dir = joinpath(Pkg.dir("Celeste"), "dat")

srand(1)
println("Loading data.")
println("Running with ", length(workers()), " processors.")

S = 100
blob, mp, body, tiled_blob =
  SampleData.gen_n_body_dataset(S, tile_width=10);

println("Calculating ELBO.")

# do a trial run first, so we don't profile/time compling the code
@time ElboDeriv.elbo(tiled_blob, mp);

# let's time it without any overhead from profiling
@time ElboDeriv.elbo(tiled_blob, mp);

# profile the code
Profile.init(10^8, 0.001)
@profile elbo = ElboDeriv.elbo(tiled_blob, mp)
Profile.print(format=:flat)
