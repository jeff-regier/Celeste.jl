#!/usr/bin/env julia

import Celeste.DeterministicVI: elbo

include(string(Pkg.dir("Celeste"), "/test/Synthetic.jl"))
include(string(Pkg.dir("Celeste"), "/test/SampleData.jl"))

const CALC_HESS = false  # with hessian?


function main()
    srand(1)
    println("Loading data.")

    S = 100
    blob, ea, body = SampleData.gen_n_body_dataset(S, tile_width=10)

    println("Calculating ELBO.")

    # do a trial run first, so we don't profile/time compling the code
    @time elbo(ea; calculate_hessian=CALC_HESS)

    # let's time it without any overhead from profiling
    @time elbo(ea; calculate_hessian=CALC_HESS)

    # on an intel core2 Q6600 processor,
    # median runtime is consistently 24 seconds with Julia 0.4
    Profile.init(10^8, 0.001)
    Profile.clear_malloc_data()
    #@profile elbo(tiled_blob, ea, calculate_hessian=CALC_HESS)
    @profile elbo(ea; calculate_hessian=CALC_HESS)
    Profile.print(format=:flat, sortedby=:count)
end


main()
