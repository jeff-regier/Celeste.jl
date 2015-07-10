# Test the functions that move between constrained and unconstrained
# parameterizations.

using Celeste
using CelesteTypes
using Base.Test
using SampleData
using Transform

import ModelInit

function test_parameter_conversion(transform::DataTransform)
	blob, mp, body = gen_three_body_dataset()
	original_vp = deepcopy(mp.vp)

	# Check that the constrain and unconstrain operations undo each other.
	vp_free = transform.from_vp(mp.vp)
	vp2 = transform.to_vp(vp_free)

	for id in names(ids), s in 1:mp.S
		@test_approx_eq original_vp[s][ids.(id)] vp2[s][ids.(id)]
	end

	# Check conversion to and from a vector.
	omitted_ids = Array(Int64, 0)
	vp = deepcopy(mp.vp)
	x = transform.vp_to_vector(vp, omitted_ids)
	@test length(x) == length(vp_free[1]) * mp.S

	# Why is this convert necessary?
	vp2 = convert(VariationalParams, [ zeros(Float64, length(vp[1])) for s = 1:mp.S ])
	transform.vector_to_vp!(x, vp2, omitted_ids)
	for id in names(ids), s in 1:mp.S
		@test_approx_eq original_vp[s][ids.(id)] vp2[s][ids.(id)]
	end
end

for trans in [ pixel_rect_transform world_rect_transform free_transform ]
	test_parameter_conversion(trans)
end
