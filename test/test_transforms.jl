# Test the functions that move between constrained and unconstrained
# parameterizations.

using Celeste: Transform, SensitiveFloats
using Compat

"""
Generate parameters within the given bounds.
"""
function generate_valid_parameters(
    NumType::DataType, bounds_vec::Vector{ParamBounds})

    @assert NumType <: Number
    S = length(bounds_vec)
    vp = convert(VariationalParams{NumType},
                                 [ zeros(NumType, length(ids)) for s = 1:S ])
        for s=1:S, (param, constraint_vec) in bounds_vec[s]
        is_box = isa(constraint_vec, Array{ParamBox})
        if is_box
            # Box parameters.
            for ind in 1:length(getfield(ids, param))
                constraint = constraint_vec[ind]
                constraint.upper_bound == Inf ?
                    vp[s][getfield(ids, param)[ind]] = constraint.lower_bound + 1.0:
                    vp[s][getfield(ids, param)[ind]] =
                        0.5 * (constraint.upper_bound - constraint.lower_bound) +
                        constraint.lower_bound
            end
        else
            # Simplex parameters can ignore the bounds.
            param_size = size(getfield(ids, param))
            if length(param_size) == 2
                # matrix simplex
                for col in 1:param_size[2]
                    vp[s][getfield(ids, param)[:, col]] = 1 / param_size[1]
                end
            else
                # vector simplex
                vp[s][getfield(ids, param)] = 1 / length(getfield(ids, param))
            end
        end
        end

    vp
end


function test_parameter_conversion()
	blob, ea, body = gen_three_body_dataset();

	transform = get_mp_transform(ea, loc_width=1.0);

	function check_transform(transform::DataTransform, ea::ElboArgs)
		original_vp = deepcopy(ea.vp);
		ea_check = deepcopy(ea);

		# Check that the constrain and unconstrain operations undo each other.
		vp_free = transform.from_vp(ea.vp)
		transform.to_vp!(vp_free, ea_check.vp)

		for id in fieldnames(ids), s in 1:ea.S
			@test_approx_eq_eps(original_vp[s][getfield(ids, id)],
			                    ea_check.vp[s][getfield(ids, id)], 1e-6)
		end

		# Check conversion to and from a vector.
		omitted_ids = Array(Int, 0)
		vp = deepcopy(ea.vp)
		x = transform.vp_to_array(vp, omitted_ids)
		@test length(x) == length(vp_free[1]) * length(ea.active_sources)

		vp2 = generate_valid_parameters(Float64, transform.bounds)
		transform.array_to_vp!(x, vp2, omitted_ids)
		for id in fieldnames(ids), si in 1:transform.active_S
			s = transform.active_sources[si]
			@test_approx_eq_eps(original_vp[s][getfield(ids, id)], vp2[si][getfield(ids, id)], 1e-6)
		end

	end

	transform = get_mp_transform(ea, loc_width=1.0);
	check_transform(transform, ea)

	# Test transforming only active sources.
	ea1 = deepcopy(ea);
	ea1.active_sources = [1]
	transform1 = Transform.get_mp_transform(ea1)

	@assert transform1.S == ea.S
	@assert transform1.active_S == 1
	check_transform(transform1, ea1)

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

		# Test that the edges work.
		simplex_and_unsimplex(Float64[ lb, 1 - lb ], simplex_box)
		simplex_and_unsimplex(Float64[ 1 - lb, lb ], simplex_box)

		# Test the scaling
		unscaled_simplex_box = Transform.SimplexBox(lb, 1.0, length(param))
		@test_approx_eq(
			Transform.unsimplexify_parameter(param, simplex_box),
			this_scale * Transform.unsimplexify_parameter(param, unscaled_simplex_box))

		# Test the bound checking
		@test_throws(AssertionError,
			Transform.unsimplexify_parameter([ lb - 1e-6, 1 - lb + 1e-6 ], simplex_box))
		@test_throws(AssertionError,
			Transform.unsimplexify_parameter([ 0.3, 0.8 ], simplex_box))
		@test_throws(AssertionError,
			Transform.unsimplexify_parameter([ 0.2, 0.3, 0.5 ], simplex_box))
		@test_throws(AssertionError, Transform.simplexify_parameter([ 1., 2. ], simplex_box))
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

		# Test that the edges work.
		box_and_unbox(lb, param_box)
		ub < Inf && box_and_unbox(ub, param_box)

		# Test the scaling
		unscaled_param_box = Transform.ParamBox(lb, ub, 1.0)
		@test_approx_eq(
			Transform.unbox_parameter(param, param_box),
			this_scale * Transform.unbox_parameter(param, unscaled_param_box))

		# Test the bound checking
		@test_throws AssertionError Transform.unbox_parameter(lb - 1.0, param_box)
		ub < Inf &&
			@test_throws AssertionError Transform.unbox_parameter(ub + 1.0, param_box)
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


function test_enforce_bounds()
	blob, ea, three_bodies = gen_three_body_dataset();
	transform = get_mp_transform(ea);

	ea.vp[1][ids.a[1]] = transform.bounds[1][:a][1].lower_bound - 0.00001
	ea.vp[2][ids.r1[1]] = transform.bounds[2][:r1][1].lower_bound - 1.0
	ea.vp[2][ids.r1[2]] = transform.bounds[2][:r1][1].upper_bound + 1.0
	ea.vp[3][ids.k[1, 1]] = transform.bounds[3][:k][1, 1].lower_bound - 0.00001

	@test_throws AssertionError transform.from_vp(ea.vp)
	Transform.enforce_bounds!(ea, transform)

	# Check that it now works and all values are finite.
	x_trans = transform.from_vp(ea.vp)
	for s = 1:ea.S
		@test !any(Bool[ isinf(x) for x in x_trans[1] ])
	end

	# Test with only one active source.
	sa = 2
	ea.active_sources = [sa]
	transform = get_mp_transform(ea);

	ea.vp[sa][ids.a[1]] = transform.bounds[1][:a][1].lower_bound - 0.00001
	ea.vp[sa][ids.r1[1]] = transform.bounds[1][:r1][1].lower_bound - 1.0
	ea.vp[sa][ids.r1[2]] = transform.bounds[1][:r1][1].upper_bound + 1.0
	ea.vp[sa][ids.k[1, 1]] = transform.bounds[1][:k][1, 1].lower_bound - 0.00001

	@test_throws AssertionError transform.from_vp(ea.vp)
	Transform.enforce_bounds!(ea, transform)

	# Check that it now works and all values are finite.
	x_trans = transform.from_vp(ea.vp)
	for s = 1:ea.S
		@test !any(Bool[ isinf(x) for x in x_trans[1] ])
	end

end


test_transform_box_functions()
test_parameter_conversion()
test_transform_simplex_functions()
test_basic_transforms()
test_enforce_bounds()
