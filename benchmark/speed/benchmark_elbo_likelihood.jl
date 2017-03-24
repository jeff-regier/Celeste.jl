#!/usr/bin/env julia

import Celeste.DeterministicVI
import Celeste.DeterministicVI: ElboIntermediateVariables, elbo_likelihood

include(joinpath(Pkg.dir("Celeste"), "test", "SampleData.jl"))

import SampleData


function benchmark_elbo_likelihood()
    ea, vp, catalog = SampleData.gen_sample_star_dataset()

    # Warm up---this compiles the code
    DeterministicVI.elbo_likelihood(ea, vp)

    # clear allocations in case julia is running with --track-allocations=user
    Profile.clear_malloc_data()

    if isempty(ARGS)
        @time DeterministicVI.elbo_likelihood(ea, vp)
    elseif ARGS[1] == "--profile"
        Profile.init(delay=0.0001)
        @profile DeterministicVI.elbo_likelihood(ea, vp)
        Profile.print(format=:flat, sortedby=:count)
    end
end


benchmark_elbo_likelihood()
