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


println("Running transform tests.")


# using Transform.ParamBox
# using Transform.SimplexBox
# using Transform.ParamBounds
# loc_width = 1.5e-3

function test_parameter_conversion()
	blob, mp, body = gen_three_body_dataset();

	transform = get_mp_transform(mp, loc_width=1.0);

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


function test_identity_transform()
	blob, mp, three_bodies = gen_three_body_dataset();
	omitted_ids = Int64[];
	kept_ids = setdiff(1:length(ids_free), omitted_ids);

	transform = Transform.get_identity_transform(length(ids), mp.S);
	@test_approx_eq reduce(hcat, mp.vp) reduce(hcat, transform.from_vp(mp.vp))
	@test_approx_eq reduce(hcat, mp.vp) reduce(hcat, transform.to_vp(mp.vp))
	xs = transform.vp_to_array(mp.vp, omitted_ids);
	@test_approx_eq  xs reduce(hcat, mp.vp)
	vp_new = deepcopy(mp.vp);
	transform.array_to_vp!(xs, vp_new, omitted_ids);
	@test_approx_eq reduce(hcat, vp_new) reduce(hcat, mp.vp)

	sf = zero_sensitive_float(CanonicalParams, Float64, mp.S);
	sf.d = rand(length(ids), mp.S)
	sf_new = transform.transform_sensitive_float(sf, mp);
	@test_approx_eq sf_new.v sf.v
	@test_approx_eq sf_new.d sf.d
	[ @test_approx_eq sf_new.h[s] sf.h[s] for s=1:mp.S]
end


function test_transform_simplex_functions()
	function simplex_and_unsimplex{NumType <: Number}(
			param::Vector{NumType}, simplex_box::Transform.SimplexBox)

		param_free = Transform.unsimplexify_parameter(param, simplex_box)
		new_param = Transform.simplexify_parameter(param_free, simplex_box)
		@test_approx_eq param new_param
	end

	for this_scale = [ 1.0, 2.0 ], lb = [0.1, 0.0 ]
		param = Float64[ 0.2, 0.8 ]

		simplex_box = Transform.SimplexBox(lb, this_scale, length(param))
		simplex_and_unsimplex(param, simplex_box)
		simplex_and_unsimplex([ Dual(p) for p in param], simplex_box)

		# Test that the edges work.
		simplex_and_unsimplex(Float64[ lb, 1 - lb ], simplex_box)
		simplex_and_unsimplex(Float64[ 1 - lb, lb ], simplex_box)

		# Test the scaling
		unscaled_simplex_box = Transform.SimplexBox(lb, 1.0, length(param))
		@test_approx_eq(
			Transform.unsimplexify_parameter(param, simplex_box),
			this_scale * Transform.unsimplexify_parameter(param, unscaled_simplex_box))

		# Test the bound checking
		@test_throws(Exception,
			Transform.unsimplexify_parameter([ lb - 1e-6, 1 - lb + 1e-6 ], simplex_box))
		@test_throws(Exception,
			Transform.unsimplexify_parameter([ 0.3, 0.8 ], simplex_box))
		@test_throws(Exception,
			Transform.unsimplexify_parameter([ 0.2, 0.3, 0.5 ], simplex_box))
		@test_throws(Exception, Transform.simplexify_parameter([ 1., 2. ], simplex_box))
	end
end


function test_transform_box_functions()
	function box_and_unbox{NumType <: Number}(param::NumType, param_box::ParamBox)
		param_free = Transform.unbox_parameter(param, param_box)
		new_param = Transform.box_parameter(param_free, param_box)
		@test_approx_eq param new_param
	end

	for this_scale = [ 1.0, 2.0 ], lb = [-10.0, 0.1], ub = [0.5, Inf]
		#println(this_scale, " ", lb, " ", ub)
		param = 0.2
		param_box = Transform.ParamBox(lb, ub, this_scale)
		box_and_unbox(param, param_box)
		box_and_unbox(Dual(param), param_box)

		# Test that the edges work.
		box_and_unbox(lb, param_box)
		ub < Inf && box_and_unbox(ub, param_box)

		# Test the scaling
		unscaled_param_box = Transform.ParamBox(lb, ub, 1.0)
		@test_approx_eq(
			Transform.unbox_parameter(param, param_box),
			this_scale * Transform.unbox_parameter(param, unscaled_param_box))

		# Test the bound checking
		@test_throws Exception Transform.unbox_parameter(lb - 1.0, param_box)
		ub < Inf &&
			@test_throws Exception Transform.unbox_parameter(ub + 1.0, param_box)
	end
end


function test_basic_transforms()
	@test_approx_eq 0.99 Transform.logit(Transform.inv_logit(0.99))
	@test_approx_eq -6.0 Transform.inv_logit(Transform.logit(-6.0))

	@test_approx_eq(
		[ 0.99, 0.001 ], Transform.logit(Transform.inv_logit([ 0.99, 0.001 ])))
	@test_approx_eq(
		[ -6.0, 0.5 ], Transform.inv_logit(Transform.logit([ -6.0, 0.5 ])))

	z = Float64[ 2.0, 4.0 ]
	z ./= sum(z)
	x = Transform.unconstrain_simplex(z)

	@test_approx_eq Transform.constrain_to_simplex(x) z

	@test Transform.inv_logit(1.0) == Inf
	@test Transform.inv_logit(0.0) == -Inf

	@test Transform.logit(Inf) == 1.0
	@test Transform.logit(-Inf) == 0.0

	@test_approx_eq Transform.constrain_to_simplex([-Inf]) [0.0, 1.0]
	@test_approx_eq Transform.unconstrain_simplex([0.0, 1.0]) [-Inf]

	@test_approx_eq Transform.constrain_to_simplex([Inf]) [1.0, 0.0]
	@test_approx_eq Transform.unconstrain_simplex([1.0, 0.0]) [Inf]

	@test_approx_eq Transform.constrain_to_simplex([Inf, 5]) [1.0, 0.0, 0.0]
end


test_identity_transform()
test_parameter_conversion()
test_transform_simplex_functions()
test_transform_box_functions()
test_basic_transforms()
