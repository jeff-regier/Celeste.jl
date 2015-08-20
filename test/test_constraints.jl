# Test the functions that move between constrained and unconstrained
# parameterizations.

using Celeste
using CelesteTypes
using Base.Test
using SampleData
using Transform

using DualNumbers
import ModelInit


function test_transform_box_functions()
	function box_and_unbox(param, lower_bound, upper_bound)
		param_free = Transform.unbox_parameter(param, lower_bound, upper_bound)
		new_param = Transform.box_parameter(param_free, lower_bound, upper_bound)
		@test_approx_eq_eps param new_param 1e-6
	end

	box_and_unbox(1.0, -1.0, 2.0)
	box_and_unbox(1.0, -1.0, Inf)

	# Test that the edges work.
	box_and_unbox(-1.0, -1.0, 2.0)
	box_and_unbox(2.0, -1.0, 2.0)
	box_and_unbox(-1.0, -1.0, Inf)

	box_and_unbox([1.0, 1.5], -1.0, 2.0)
	box_and_unbox([1.0, 1.5], -1.0, Inf)

	box_and_unbox([1.0, 10.0], [-1.0, 9.0], [2.0, 12.0])
	box_and_unbox([1.0, 10.0], [-1.0, 9.0], [Inf, Inf])

	box_and_unbox(Dual(1.0), -1.0, 2.0)
	box_and_unbox([Dual(1.0), Dual(1.5)], -1.0, 2.0)
	box_and_unbox([Dual(1.0), Dual(10.0)], [-1.0, 9.0], [2.0, 12.0])

	# Just check that these run.  The derivatives themselves
	# will be checked elsewhere.
	Transform.unbox_derivative(1.0, 2.0, -1.0, 2.0)
	Transform.unbox_derivative(1.0, 2.0, -1.0, Inf)
	Transform.unbox_derivative(
		[1.0, 10.0], [2.0, 3.0], [-1.0, 9.0], [2.0, 12.0])
	Transform.unbox_derivative(
		[1.0, 10.0], [2.0, 3.0], [-1.0, 9.0], [Inf, Inf])
	Transform.unbox_derivative(Dual(1.0), Dual(2.0), -1.0, 2.0)
	Transform.unbox_derivative(
		[Dual(1.0), Dual(10.0)], [Dual(2.0), Dual(3.0)], [-1.0, 9.0], [2.0, 12.0])

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

	vp2 = generate_valid_parameters(Float64, transform.bounds)
	transform.vector_to_vp!(x, vp2, omitted_ids)
	for id in names(ids), s in 1:mp.S
		@test_approx_eq_eps(original_vp[s][ids.(id)], vp2[s][ids.(id)], 1e-6)
	end
end

test_parameter_conversion()
test_transform_box_functions()
