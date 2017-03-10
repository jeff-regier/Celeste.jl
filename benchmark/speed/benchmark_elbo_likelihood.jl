#!/usr/bin/env julia

import Celeste.DeterministicVI
import Celeste.DeterministicVI: ElboIntermediateVariables, elbo_likelihood

include(joinpath(Pkg.dir("Celeste"), "test", "Synthetic.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "SampleData.jl"))

import Synthetic
import SampleData


function benchmark_elbo_likelihood()
    _, ea, _ = SampleData.gen_sample_star_dataset()

    # Warm up---this compiles the code
    DeterministicVI.elbo_likelihood(ea)

    # clear allocations in case julia is running with --track-allocations=user
    Profile.clear_malloc_data()

    if isempty(ARGS)
        @time DeterministicVI.elbo_likelihood(ea)
    elseif ARGS[1] == "--profile"
        Profile.init(delay=0.0001)
        @profile DeterministicVI.elbo_likelihood(ea)
        Profile.print(format=:flat, sortedby=:count)
    end
end


benchmark_elbo_likelihood()
