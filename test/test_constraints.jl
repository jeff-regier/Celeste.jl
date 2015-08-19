# Test the functions that move between constrained and unconstrained
# parameterizations.

using Celeste
using CelesteTypes
using Base.Test
using SampleData
using Transform

import ModelInit


function test_parameter_conversion()
	blob, mp, body = gen_three_body_dataset();
	transform = get_mp_transform(mp, loc_width=1.0);
	original_vp = deepcopy(mp.vp);

	# Check that the constrain and unconstrain operations undo each other.
	vp_free = transform.from_vp(mp.vp)
	vp2 = transform.to_vp(vp_free)

	for id in names(ids), s in 1:mp.S
		@test_approx_eq_eps(original_vp[s][ids.(id)], vp2[s][ids.(id)], 1e-6)
	end

	# Check conversion to and from a vector.
	omitted_ids = Array(Int64, 0)
	vp = deepcopy(mp.vp)
	x = transform.vp_to_vector(vp, omitted_ids)
	@test length(x) == length(vp_free[1]) * mp.S

	# Generate parameters within the bounds.
	vp2 = convert(VariationalParams{Float64},
	              [ zeros(Float64, length(vp[1])) for s = 1:mp.S ])
	for s=1:mp.S
		for (param, limits) in transform.bounds[s]
	    vp2[s][ids.(param)] = 0.5 * (limits[2] - limits[1]) + limits[1]
	  end
	end

	transform.vector_to_vp!(x, vp2, omitted_ids)
	for id in names(ids), s in 1:mp.S
		@test_approx_eq_eps(original_vp[s][ids.(id)], vp2[s][ids.(id)], 1e-6)
	end
end

test_parameter_conversion()
