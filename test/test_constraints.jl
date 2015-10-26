# Test the functions that move between constrained and unconstrained
# parameterizations.

using Celeste
using CelesteTypes
using Base.Test
using SampleData
using Transform
using Compat

using DualNumbers
import ModelInit

println("Running constraint tests.")

function test_transform_box_functions()
	function box_and_unbox(param, lower_bound, upper_bound; scale=1.0)
		param_free = Transform.unbox_parameter(param, lower_bound, upper_bound, scale)
		new_param = Transform.box_parameter(param_free, lower_bound, upper_bound, scale)
		@test_approx_eq_eps param new_param 1e-6
	end

	for this_scale = [ 1.0, 2.0 ]
		box_and_unbox(1.0, -1.0, 2.0, scale=this_scale)
		box_and_unbox(1.0, -1.0, Inf, scale=this_scale)

		# Test that the edges work.
		box_and_unbox(-1.0, -1.0, 2.0, scale=this_scale)
		box_and_unbox(2.0, -1.0, 2.0, scale=this_scale)
		box_and_unbox(-1.0, -1.0, Inf, scale=this_scale)
		box_and_unbox(Dual(1.0), -1.0, 2.0, scale=this_scale)
	end

	for this_scale = (1.0, 2.0, [2.0, 3.0])
		box_and_unbox([1.0, 1.5], -1.0, 2.0, scale=this_scale)
		box_and_unbox([1.0, 1.5], -1.0, Inf, scale=this_scale)

		box_and_unbox([1.0, 10.0], [-1.0, 9.0], [2.0, 12.0], scale=this_scale)
		box_and_unbox([1.0, 10.0], [-1.0, 9.0], [Inf, Inf], scale=this_scale)

		box_and_unbox([Dual(1.0), Dual(1.5)], -1.0, 2.0, scale=this_scale)
		box_and_unbox([Dual(1.0), Dual(10.0)], [-1.0, 9.0], [2.0, 12.0],
									scale=this_scale)
	end


	# Just check that these run.  The derivatives themselves
	# will be checked elsewhere.
	this_scale = 1.0
	Transform.unbox_derivative(1.0, 2.0, -1.0, 2.0, this_scale)
	Transform.unbox_derivative(1.0, 2.0, -1.0, Inf, this_scale)
	Transform.unbox_derivative(
		[1.0, 10.0], [2.0, 3.0], [-1.0, 9.0], [2.0, 12.0], this_scale)
	Transform.unbox_derivative(
		[1.0, 10.0], [2.0, 3.0], [-1.0, 9.0], [Inf, Inf], this_scale)
	Transform.unbox_derivative(Dual(1.0), Dual(2.0), -1.0, 2.0, this_scale)
	Transform.unbox_derivative(
		[Dual(1.0), Dual(10.0)], [Dual(2.0), Dual(3.0)], [-1.0, 9.0], [2.0, 12.0],
		this_scale)

	# Check that the scaling is working.
	@test_approx_eq_eps(Transform.unbox_parameter(1.0, -1.0, 2.0, 2.0),
											Transform.unbox_parameter(1.0, -1.0, 2.0, 1.0) * 2.0,
											1e-6)
	@test_approx_eq_eps(Transform.unbox_parameter(1.0, -1.0, Inf, 2.0),
											Transform.unbox_parameter(1.0, -1.0, Inf, 1.0) * 2.0,
											1e-6)
	@test_approx_eq_eps(
		Transform.unbox_derivative(1.0, 2.0, -1.0, 2.0, 2.0),
	  Transform.unbox_derivative(1.0, 2.0, -1.0, 2.0, 1.0) * 0.5,
		1e-6)
	@test_approx_eq_eps(
		Transform.unbox_derivative(1.0, 2.0, -1.0, Inf, 2.0),
	  Transform.unbox_derivative(1.0, 2.0, -1.0, Inf, 1.0) * 0.5,
		1e-6)

	# Check the bounds checking errors.
	@test_throws Exception Transform.unbox_parameter(1.0, 2.0, 3.0)
	@test_throws Exception Transform.unbox_parameter([1.0, 1.5], 2.0, 3.0)
	@test_throws Exception Transform.unbox_parameter(
		[1.0, 10.0], [2.0, 3.0], [9.0, 12.0])

	# Check that mixed bound types throws an error.
	@test_throws Exception Transform.unbox_parameter(
		[1.0, 10.0], [-1.0, 9.0], [2.0, Inf])
	@test_throws Exception Transform.box_parameter(
		[1.0, 10.0], [-1.0, 9.0], [2.0, Inf])
	@test_throws Exception Transform.unbox_derivative(
		[1.0, 10.0], [2.0, 3.0], [-1.0, 9.0], [2.0, Inf])
end


function test_parameter_conversion()
	blob, mp, body = gen_three_body_dataset();

	function check_transform(transform::DataTransform, mp::ModelParams)
		original_vp = deepcopy(mp.vp);
		mp_check = deepcopy(mp);

		# Check that the constrain and unconstrain operations undo each other.
		vp_free = transform.from_vp(mp.vp)
		transform.to_vp!(vp_free, mp_check.vp)

		for id in fieldnames(ids), s in 1:mp.S
			@test_approx_eq_eps(original_vp[s][ids.(id)],
			                    mp_check.vp[s][ids.(id)], 1e-6)
		end

		# Check conversion to and from a vector.
		omitted_ids = Array(Int64, 0)
		vp = deepcopy(mp.vp)
		x = transform.vp_to_array(vp, omitted_ids)
		@test length(x) == length(vp_free[1]) * length(mp.active_sources)

		vp2 = generate_valid_parameters(Float64, transform.bounds)
		transform.array_to_vp!(x, vp2, omitted_ids)
		for id in fieldnames(ids), si in 1:transform.active_S
			s = transform.active_sources[si]
			@test_approx_eq_eps(original_vp[s][ids.(id)], vp2[si][ids.(id)], 1e-6)
		end
	end

	transform = get_mp_transform(mp, loc_width=1.0);
	check_transform(transform, mp)

	# Test transforming only active sources.
	mp1 = deepcopy(mp);
	mp1.active_sources = [1]
	transform1 = Transform.get_mp_transform(mp1)

	@assert transform1.S == mp.S
	@assert transform1.active_S == 1
	check_transform(transform1, mp1)

end

test_transform_box_functions()
test_parameter_conversion()
