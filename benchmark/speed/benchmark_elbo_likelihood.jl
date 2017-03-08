#!/usr/bin/env julia

import ForwardDiff: Dual

import Celeste.DeterministicVI
import Celeste.DeterministicVI: ElboIntermediateVariables, elbo_likelihood

include(joinpath(Pkg.dir("Celeste"), "test", "Synthetic.jl"))
include(joinpath(Pkg.dir("Celeste"), "test", "SampleData.jl"))

import Synthetic
import SampleData


function benchmark_elbo_likelihood()
    ea, vp, catalog = SampleData.gen_sample_star_dataset()

    vp_dual = Vector{Dual{1, Float64}}[similar(src, Dual{1, Float64}) for src in vp]
    for i in 1:length(vp)
        vp_dual[i] = vp[i]
    end

    elbo_vars = ElboIntermediateVariables(Dual{1, Float64}, ea.S, ea.Sa, true, false)

    # Warm up---this compiles the code
    DeterministicVI.elbo_likelihood(ea, vp_dual, elbo_vars)

    # clear allocations in case julia is running with --track-allocations=user
    Profile.clear_malloc_data()

    if isempty(ARGS)
        @time DeterministicVI.elbo_likelihood(ea, vp_dual, elbo_vars)
    elseif ARGS[1] == "--profile"
        Profile.init(delay=0.01)
        @profile DeterministicVI.elbo_likelihood(ea, vp_dual, elbo_vars)
        Profile.print(format=:flat, sortedby=:count)
    end
end


benchmark_elbo_likelihood()
