#!/usr/bin/env julia

using Celeste
using CelesteTypes
using Dates

import SampleData
import ElboJuMP

using JuMP


function test_jump_optimization()
    # The maximum width and height of the testing image in pixels.
    # Change these to test sub-images of different sizes.
    const max_size = 1

    # Load some simulated data.  blob contains the image data, and
    # mp is the parameter values.  three_bodies is not used.
    # For now, treat these as global constants accessed within the expressions
    const blob, mp, three_bodies = SampleData.gen_three_body_dataset();

    # Reduce the size of the images for debugging
    for b in 1:CelesteTypes.B
        this_height = min(blob[b].H, max_size)
        this_width = min(blob[b].W, max_size)
        blob[b].H = this_height
        blob[b].W = this_width
        blob[b].pixels = blob[b].pixels[1:this_height, 1:this_width] 
    end

    jump_m, jump_elbo = ElboJuMP.build_jump_model(blob, mp);

    @setNLObjective(jump_m, Max, jump_elbo)
    println("calling `solve` for a $(max_size^2)-pixel image...")
    status = solve(jump_m)
    println("Objective value: ", getObjectiveValue(jump_m))
end


test_jump_optimization()


