#!/usr/bin/env julia

using Celeste
using Celeste: CelesteTypes
using Dates

import Celeste: SampleData, ElboJuMP, ElboDeriv

import MathProgBase


function compare_jump_speed()
    # The maximum width and height of the testing image in pixels.
    # Change these to test sub-images of different sizes.
    const max_size = 20

    # Load some simulated data.  images contains the image data, and
    # mp is the parameter values.  three_bodies is not used.
    # For now, treat these as global constants accessed within the expressions
    const images, mp, three_bodies = SampleData.gen_three_body_dataset();

    # Reduce the size of the images for debugging
    for b in 1:CelesteTypes.B
        this_height = min(images[b].H, max_size)
        this_width = min(images[b].W, max_size)
        images[b].H = this_height
        images[b].W = this_width
        images[b].pixels = images[b].pixels[1:this_height, 1:this_width] 
    end

    jump_m, jump_elbo = ElboJuMP.build_jump_model(images, mp);
    d = JuMP.NLPEvaluator(jump_m)
    @time MathProgBase.initialize(d, [:Grad,:ExprGraph])
    obj_expr = MathProgBase.obj_expr(d)

    grad_out = zeros(length(jump_m.colVal))

    # Run it more than once so we don't capture compilation time.
    for iter = 1:5
        println("trial $iter.")

        println("computing manually...")
        manual_time = @elapsed manual_v = ElboDeriv.elbo_likelihood(images, mp).v
        @show manual_time

        println("computing with JuMP...")
        tic()
        objective_time = @elapsed jump_v = MathProgBase.eval_f(d, jump_m.colVal)
        @show jump_time
        grad_time = @elapsed MathProgBase.eval_g(d, grad_out, jump_m.colVal)
        @show grad_time
        jump_time = toc()

        println("objective values: $manual_v (manual) vs. $jump_v (jump)")
        @printf("JuMP overhead: %.1fX\n\n", jump_time / manual_time)
    end
end


compare_jump_speed()

