#!/usr/bin/env julia

using Celeste
using CelesteTypes
using Dates

import SampleData
import ElboJuMP

using JuMP


function test_jump_optimization()
    # Load some simulated data.  blob contains the image data, and
    # mp is the parameter values.  three_bodies is not used.
    # For now, treat these as global constants accessed within the expressions
    const blob, mp, three_bodies = SampleData.gen_three_body_dataset();

    jump_m, jump_elbo = ElboJuMP.build_jump_model(blob, mp);

    @setNLObjective(jump_m, Max, jump_elbo)
    num_pixels = sum([img.H * img.W for img in blob])
    println("calling `solve` for a $(num_pixels)-pixel image...")
    status = solve(jump_m)
    println("Objective value: ", getObjectiveValue(jump_m))
end


test_jump_optimization()


