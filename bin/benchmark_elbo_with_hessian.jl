#!/usr/bin/env julia

using Celeste
using CelesteTypes

import Synthetic
import ModelInit
import SampleData

const dat_dir = joinpath(Pkg.dir("Celeste"), "dat")

srand(1)
println("Loading data.")

S = 4
blob, mp, body, tiled_blob =
  SampleData.gen_n_body_dataset(S, tile_width=10);

println("Calculating ELBO.")

# do a trial run first, so we don't profile/time compling the code
@time elbo = ElboDeriv.elbo(tiled_blob, mp, calculate_hessian=true);

# let's time it without any overhead from profiling
@time elbo = ElboDeriv.elbo(tiled_blob, mp, calculate_hessian=true);
@time elbo = ElboDeriv.elbo(tiled_blob, mp, calculate_hessian=true);

Profile.init(10^8, 0.001)
Profile.clear_malloc_data()
#@profile elbo = ElboDeriv.elbo(tiled_blob, mp, calculate_hessian=false);
@profile elbo = ElboDeriv.elbo_likelihood(tiled_blob, mp, calculate_hessian=false);
Profile.print(format=:flat)
