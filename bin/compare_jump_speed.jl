#!/usr/bin/env julia

using Celeste
using CelesteTypes
using Dates

import SampleData
import ElboJuMP


function compare_jump_speed()
    # The maximum width and height of the testing image in pixels.
    # Change these to test sub-images of different sizes.
    const max_size = 20

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
    obj_expr = @ReverseDiffSparse.processNLExpr jump_elbo
    fg = ReverseDiffSparse.genfgrad_simple(obj_expr)
    grad_out = zeros(length(jump_m.colVal))

    # @code_warntype  ReverseDiffSparse.getvalue(jump_elbo, jump_m.colVal)

    # Run it more than once so we don't capture compilation time.
    for iter = 1:5
        println("trial $iter.")

        println("computing manually...")
        manual_time = @elapsed manual_v = ElboDeriv.elbo_likelihood(blob, mp).v
        @show manual_time

        println("computing with JuMP...")
        jump_time = @elapsed jump_v = ReverseDiffSparse.getvalue(jump_elbo, jump_m.colVal)
        @show jump_time
        grad_time = @elapsed fg(jump_m.colVal, grad_out)
        @show grad_time


        println("objective values: $manual_v (manual) vs. $jump_v (jump)")
        @printf("JuMP overhead: %.1fX\n\n", jump_time / manual_time)
    end
end


compare_jump_speed()

