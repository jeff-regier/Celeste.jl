# A minimal exmample of the ElboJuMP code.

using Celeste
using CelesteTypes
using Base.Test
using Dates

import SampleData
import MinimalElboJuMP


function minimial_jump_test()
    # The maximum width and height of the testing image in pixels.
    # Change these to test sub-images of different sizes.  Note
    # that after changing you have to re-run all the code below,
    # since it is used both to modify the blob object and,
    # in turn, set constants in MinimalElboJuMP.jl.
    max_size = 10

    # Load some simulated data.  blob contains the image data, and
    # mp is the parameter values.  three_bodies is not used.
    # For now, treat these as global constants accessed within the expressions
    blob, mp, three_bodies = SampleData.gen_three_body_dataset();

    # Reduce the size of the images for debugging
    for b in 1:CelesteTypes.B
        this_height = min(blob[b].H, max_size)
        this_width = min(blob[b].W, max_size)
        blob[b].H = this_height
        blob[b].W = this_width
        blob[b].pixels = blob[b].pixels[1:this_height, 1:this_width] 
    end

    jump_m, jump_elbo = MinimalElboJuMP.build_jump_model(blob, mp)

    # Run it twice to make sure that we don't capture compilation time.
    total_time = now()
    jump_time = 0.0
    celeste_time = 0.0
    for iter = 1:2
        print("iter ", iter, "\n")
        # Compare the times.  This comparison is not particularly meaningful,
        # since celeste is doing much more than the minimal JuMP example.
        print("Celeste.\n")
        celeste_time = @elapsed ElboDeriv.elbo_likelihood(blob, mp).v
        print("JuMP.\n")
        jump_time = @elapsed ReverseDiffSparse.getvalue(jump_elbo, jump_m.colVal)
        print(max_size, ": ", jump_time / celeste_time, "\n")
    end
    total_time = now() - total_time

    print(max_size, ": ", jump_time / celeste_time, "\n")
end


minimial_jump_test()

