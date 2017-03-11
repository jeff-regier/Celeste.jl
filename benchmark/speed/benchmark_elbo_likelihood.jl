#!/usr/bin/env julia

import Celeste.DeterministicVI
import Celeste.DeterministicVI: ElboIntermediateVariables, elbo_likelihood

include(joinpath(Pkg.dir("Celeste"), "test", "Synthetic.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "SampleData.jl"))

import Synthetic
import SampleData


function benchmark_elbo_likelihood()
    ea, vp, catalog = SampleData.gen_sample_star_dataset()
    elbo_vars = ElboIntermediateVariables(Float64, ea.psf_K, ea.S, ea.Sa, true, true)

    # Warm up---this compiles the code
    DeterministicVI.elbo_likelihood(ea, vp, elbo_vars)

    # clear allocations in case julia is running with --track-allocations=user
    Profile.clear_malloc_data()

    if isempty(ARGS)
        @time DeterministicVI.elbo_likelihood(ea, vp, elbo_vars)
    elseif ARGS[1] == "--profile"
        Profile.init(delay=0.0001)
        @profile DeterministicVI.elbo_likelihood(ea, vp, elbo_vars)
        Profile.print(format=:flat, sortedby=:count)
    end
end


benchmark_elbo_likelihood()
