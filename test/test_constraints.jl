# Test the functions that move between constrained and unconstrained
# parameterizations.

using Celeste
using CelesteTypes
using Base.Test
using SampleData
using Constrain

import ModelInit

function test_parameter_conversion()
	blob, mp, body = gen_sample_star_dataset()

	# Check that the constrain and unconstrain operations undo each other.
	vp2 = deepcopy(mp.vp)
	vp_free = unconstrain_vp(vp2)
	constrain_vp!(vp_free, vp2)
	for (id in names(ids)), s in 1:mp.S
		@test_approx_eq mp.vp[s][ids.(id)] vp2[s][ids.(id)]
	end

	# Check that the constrain and unconstrain operations undo each other.
	vp2 = deepcopy(mp.vp)
	vp_free = rect_unconstrain_vp(vp2)
	rect_constrain_vp!(vp_free, vp2)
	for (id in names(ids)), s in 1:mp.S
		@test_approx_eq mp.vp[s][ids.(id)] vp2[s][ids.(id)]
	end
end




function test_sensitive_float_conversion()
	function chi_function(vp::VariationalParams)
		const p_true = 0.7

		# A float that is sensitive to chi
		val = zero_sensitive_float([ 1 ], [ all_params ] )
		val.v = (0.1 * (vp[1][ids.chi[2]] - p_true) ^ 2 +
			     0.2 * (vp[1][ids.chi[1]] - (1 - p_true)) ^ 2)

		val.d[ ids.chi[1] ] = 2 * 0.2 * (vp[1][ids.chi[1]] - (1 - p_true))
		val.d[ ids.chi[2] ] = 2 * 0.1 * (vp[1][ids.chi[2]] - p_true)

		val
	end

	epsilon = 1e-6

	mp = empty_model_params(1)
	mp.vp[1][ids.chi] = [ 0.5, 0.5 ]
	vp_rect = rect_unconstrain_vp(mp.vp)
	vp_free = unconstrain_vp(mp.vp)

	val = chi_function(mp.vp)
	rect_val = rect_unconstrain_sensitive_float(val, mp)
	free_val = unconstrain_sensitive_float(val, mp)

	mp_perturb = deepcopy(mp)
	mp_perturb.vp[1][ids.chi] = [ 0.5 + epsilon, 0.5 - epsilon ]
	vp_perturb_rect = rect_unconstrain_vp(mp_perturb.vp)
	vp_perturb_free = unconstrain_vp(mp_perturb.vp)

	perturb_val = chi_function(mp_perturb.vp)
	perturb_rect_val = rect_unconstrain_sensitive_float(perturb_val, mp)
	perturb_free_val = unconstrain_sensitive_float(perturb_val, mp)

	rect_delta = (vp_perturb_rect[1][ids_free.chi_free] - vp_rect[1][ids_free.chi_free])[1]
	free_delta = (vp_perturb_free[1][ids_free.chi_free] - vp_free[1][ids_free.chi_free])[1]

	rect_numeric_deriv = (perturb_val.v - val.v) / rect_delta
	free_numeric_deriv = (perturb_val.v - val.v) / free_delta

	@test_approx_eq_eps rect_numeric_deriv  perturb_rect_val.d[ids_free.chi_free][1] epsilon
	@test_approx_eq_eps free_numeric_deriv  perturb_free_val.d[ids_free.chi_free][1] epsilon
end

test_parameter_conversion()
test_sensitive_float_conversion()