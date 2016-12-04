#!/usr/bin/env julia

import Celeste.DeterministicVI: elbo, ElboArgs
using DiffBase

include(string(Pkg.dir("Celeste"), "/test/Synthetic.jl"))
include(string(Pkg.dir("Celeste"), "/test/SampleData.jl"))


function main()
    srand(1)
    println("Loading data.")

    S = 100
    blob, ea, body = SampleData.gen_n_body_dataset(S)
    ea = ElboArgs(ea.images, ea.vp, ea.patches, ea.active_sources;
                  psf_K=ea.psf_K,
                  calculate_gradient=true,
                  calculate_hessian=false)

    param_length = length(Celeste.Model.CanonicalParams)
    kl_source = Celeste.SensitiveFloats.SensitiveFloat{Float64}(param_length, 1, true, false)
    kl_grad = DiffBase.GradientResult(zeros(param_length))

    println("Warm-up / compiling.")
    # do a trial run first, so we don't profile/time compling the code
    elbo(ea, kl_source, kl_grad, nothing)
    Profile.clear_malloc_data()

    println("Calculating ELBO and gradient.")
    if isempty(ARGS)
        # let's time it without any overhead from profiling
        @time elbo(ea, kl_source, kl_grad, nothing)
    elseif ARGS[1] == "--profile"
        Profile.init(delay=0.001)
        @profile elbo(ea, kl_source, kl_grad, nothing)
        Profile.print(format=:flat, sortedby=:count)
    end
end


main()
