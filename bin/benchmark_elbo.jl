#!/usr/bin/env julia

import Celeste: SampleData, ElboDeriv

const CALC_HESS = false  # with hessian?

srand(1)
println("Loading data.")

S = 100
blob, mp, body, tiled_blob =
  SampleData.gen_n_body_dataset(S, tile_width=10);

println("Calculating ELBO.")

# do a trial run first, so we don't profile/time compling the code
@time elbo = ElboDeriv.elbo(tiled_blob, mp, calculate_hessian=CALC_HESS);

# let's time it without any overhead from profiling
@time elbo = ElboDeriv.elbo(tiled_blob, mp, calculate_hessian=CALC_HESS);

# on a intel core2 Q6600 processor,
# median runtime is consistently 27 seconds with Julia 0.3
# median runtime is consistently 24 seconds with Julia 0.4
Profile.init(10^8, 0.001)
Profile.clear_malloc_data()
#@profile elbo = ElboDeriv.elbo(tiled_blob, mp, calculate_hessian=CALC_HESS);
@profile elbo = ElboDeriv.elbo_likelihood(tiled_blob, mp, calculate_hessian=CALC_HESS);
Profile.print(format=:flat)
