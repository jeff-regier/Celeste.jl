# Test the functions that move between constrained and unconstrained
# parameterizations.

using Celeste: Transform, SensitiveFloats
using DerivativeTestUtils

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
                constraint.ub == Inf ?
                    vp[s][getfield(ids, param)[ind]] = constraint.lb + 1.0:
                    vp[s][getfield(ids, param)[ind]] =
                        0.5 * (constraint.ub - constraint.lb) +
                        constraint.lb
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

    transform = get_mp_transform(ea.vp, ea.active_sources, loc_width=1.0);

    function check_transform(transform::DataTransform, ea::ElboArgs)
        original_vp = deepcopy(ea.vp);
        ea_check = deepcopy(ea);

        # Check that the constrain and unconstrain operations undo each other.
        vp_free = Transform.from_vp(transform, ea.vp)
        Transform.to_vp!(transform, vp_free, ea_check.vp)

        for id in fieldnames(ids), s in 1:ea.S
            @test isapprox(original_vp[s][getfield(ids, id)],
                           ea_check.vp[s][getfield(ids, id)], atol=1e-6)
        end

        # Check conversion to and from a vector.
        omitted_ids = Vector{Int}(0)
        kept_ids = setdiff(1:length(UnconstrainedParams), omitted_ids)
        vp = deepcopy(ea.vp)
        x = Transform.vp_to_array(transform, vp, omitted_ids)
        @test length(x) == length(vp_free[1]) * length(ea.active_sources)

        vp2 = generate_valid_parameters(Float64, transform.bounds)
        Transform.array_to_vp!(transform, x, vp2, kept_ids)

        for id in fieldnames(ids), si in eachindex(transform.active_sources)
            s = transform.active_sources[si]
            @test isapprox(original_vp[s][getfield(ids, id)], vp2[si][getfield(ids, id)], atol=1e-6)
        end

    end

    transform = get_mp_transform(ea.vp, ea.active_sources, loc_width=1.0);
    check_transform(transform, ea)

    # Test transforming only active sources.
    ea1 = deepcopy(ea);
    ea1.active_sources = [1]
    transform1 = Transform.get_mp_transform(ea1.vp, ea1.active_sources)

    @assert transform1.S == ea.S
    @assert length(transform1.active_sources) == 1
    check_transform(transform1, ea1)

end


function test_transform_simplex_functions()
    function simplex_and_unsimplex{NumType <: Number}(
            param::Vector{NumType}, simplex_box::Transform.SimplexBox)

        param_free = Transform.unsimplexify_parameter(param, simplex_box)
        new_param = Transform.simplexify_parameter(param_free, simplex_box)
        @test param ≈ new_param
    end

    for this_scale = [ 1.0, 2.0 ], lb = [0.1, 0.0 ]
        param = Float64[ 0.2, 0.8 ]

        simplex_box = Transform.SimplexBox(lb, this_scale, length(param))
        simplex_and_unsimplex(param, simplex_box)

        # Test the scaling
        unscaled_simplex_box = Transform.SimplexBox(lb, 1.0, length(param))
        @test isapprox(
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
        @test param ≈ new_param
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
        @test isapprox(
            Transform.unbox_parameter(param, param_box),
            this_scale * Transform.unbox_parameter(param, unscaled_param_box))

        # Test the bound checking
        @test_throws AssertionError Transform.unbox_parameter(lb - 1.0, param_box)
        ub < Inf &&
            @test_throws AssertionError Transform.unbox_parameter(ub + 1.0, param_box)
    end
end


function test_basic_transforms()
    @test 0.99 ≈ Transform.logit(Transform.inv_logit(0.99))
    @test -6.0 ≈ Transform.inv_logit(Transform.logit(-6.0))

    @test isapprox(
        [ 0.99, 0.001 ], Transform.logit.(Transform.inv_logit.([ 0.99, 0.001 ])))
    @test isapprox(
        [ -6.0, 0.5 ], Transform.inv_logit.(Transform.logit.([ -6.0, 0.5 ])))

    z = Float64[ 2.0, 4.0 ]
    z ./= sum(z)
    x = Transform.unconstrain_simplex(z)

    @test Transform.constrain_to_simplex(x) ≈ z

    @test Transform.inv_logit(1.0) == Inf
    @test Transform.inv_logit(0.0) == -Inf

    @test Transform.logit(Inf) == 1.0
    @test Transform.logit(-Inf) == 0.0

    @test Transform.constrain_to_simplex([Inf, 5])  ≈ [1.0, 0.0, 0.0]

    @test sum(Transform.constrain_to_simplex([Inf, Inf])) ≈ 1.0 # Make sure that it's a simplex when there is more than one Inf

    @test sum(Transform.constrain_to_simplex([709.0, 709.0, 709.0])) ≈ 1.0 # sum overflows
    @test sum(Transform.constrain_to_simplex([710.0, 710.0, 710.0])) ≈ 1.0 # each element overflows
    @test sum(Transform.constrain_to_simplex([88.0f0, 88.0f0, 88.0f0])) ≈ 1.0f0 # sum overflows
    @test sum(Transform.constrain_to_simplex([89.0f0, 89.0f0, 89.0f0])) ≈ 1.0f0 # each element overflows
end


function test_enforce_bounds()
  blob, ea_original, three_bodies = gen_three_body_dataset();
  transform = get_mp_transform(ea_original.vp, ea_original.active_sources);

  ea = deepcopy(ea_original);
  ea.vp[1][ids.a[1, 1]] = transform.bounds[1][:a][1].lb - 0.00001
  ea.vp[2][ids.r1[1]] = transform.bounds[2][:r1][1].lb - 1.0
  ea.vp[2][ids.r1[2]] = transform.bounds[2][:r1][1].ub + 1.0
  ea.vp[3][ids.k[1, 1]] = transform.bounds[3][:k][1].lb - 0.00001

  @test_throws AssertionError Transform.from_vp(transform, ea.vp)
  Transform.enforce_bounds!(ea.vp, ea.active_sources, transform)

  # Check that it now works and all values are finite.
  x_trans = Transform.from_vp(transform, ea.vp)
  for s = 1:ea.S
      @test !any(Bool[ isinf(x) for x in x_trans[1] ])
  end


  # Test a corner case of the simplex bounds to make sure that normalizing
  # doesn't violate the minimization constraints
  ea = deepcopy(ea_original);
  constraint = transform.bounds[1][:a][1]
  ea.vp[1][ids.a[1, 1]] = transform.bounds[1][:a][1].lb - 0.00001
  ea.vp[1][ids.a[2, 1]] = 100
  @test_throws AssertionError Transform.from_vp(transform, ea.vp)
  Transform.enforce_bounds!(ea.vp, ea.active_sources, transform)
  # Check that it runs without an error now
  Transform.from_vp(transform, ea.vp)


  # Test with only one active source.
  ea = deepcopy(ea_original);
  sa = 2
  ea.active_sources = [sa]
  transform = get_mp_transform(ea.vp, ea.active_sources);

  ea.vp[sa][ids.a[1, 1]] = transform.bounds[1][:a][1].lb - 0.00001
  ea.vp[sa][ids.r1[1]] = transform.bounds[1][:r1][1].lb - 1.0
  ea.vp[sa][ids.r1[2]] = transform.bounds[1][:r1][1].ub + 1.0
  ea.vp[sa][ids.k[1, 1]] = transform.bounds[1][:k][1].lb - 0.00001

  @test_throws AssertionError Transform.from_vp(transform, ea.vp)
  Transform.enforce_bounds!(ea.vp, ea.active_sources, transform)

  # Check that it now works and all values are finite.
  x_trans = Transform.from_vp(transform, ea.vp)
  for s = 1:ea.S
      @test !any(Bool[ isinf(x) for x in x_trans[1] ])
  end
end


function test_omitted_ids()
    images, ea, body = true_star_init()
    DeterministicVI.maximize_f(DeterministicVI.elbo, ea, omitted_ids=[1,2])
end


test_omitted_ids()
test_transform_box_functions()
test_parameter_conversion()
test_transform_simplex_functions()
test_basic_transforms()
test_enforce_bounds()
