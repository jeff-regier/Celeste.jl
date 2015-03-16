# A minimal exmample of the ElboJuMP code.

using Celeste
using CelesteTypes
using Base.Test
using Dates

import SampleData

# The maximum width and height of the testing image in pixels.
# Change these to test sub-images of different sizes.
max_size = 100

# Load some simulated data.  blobs contains the image data, and
# mp is the parameter values.  three_bodies is not used.
# For now, treat these as global constants accessed within the expressions
blobs, mp, three_bodies = SampleData.gen_three_body_dataset();

# Reduce the size of the images for debugging
for b in 1:CelesteTypes.B
	this_height = min(blobs[b].H, max_size)
	this_width = min(blobs[b].W, max_size)
	blobs[b].H = this_height
	blobs[b].W = this_width
	blobs[b].pixels = blobs[b].pixels[1:this_height, 1:this_width] 
end

# Populate the brightness arrays using the original Celeste code
E_l_a = [ ElboDeriv.SourceBrightness(mp.vp[s]).E_l_a[b, a].v
          for s=1:mp.S, b=1:CelesteTypes.B, a=1:CelesteTypes.I ];

E_ll_a = [ ElboDeriv.SourceBrightness(mp.vp[s]).E_ll_a[b, a].v
           for s=1:mp.S, b=1:CelesteTypes.B, a=1:CelesteTypes.I ];

# Define the minimal JuMP example and test the times.
include(joinpath(Pkg.dir("Celeste"), "src/MinimalElboJuMP.jl"))
SetJuMPParameters(mp)

# Run it twice to make sure that we don't capture compilation time.
total_time = now()
jump_time = 0.0
celeste_time = 0.0
for iter = 1:2
	print("iter ", iter, "\n")
	# Compare the times.  This comparison is not particularly meaningful,
	# since celeste is doing much more than the minimal JuMP example.
	print("Celeste.\n")
	celeste_time = @elapsed ElboDeriv.elbo_likelihood(blobs, mp).v
	print("JuMP.\n")
	jump_time = @elapsed ReverseDiffSparse.getvalue(elbo_log_likelihood, celeste_m.colVal)
	print(max_size, ": ", jump_time / celeste_time, "\n")
end
total_time = now() - total_time

print(max_size, ": ", jump_time / celeste_time, "\n")