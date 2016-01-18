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




using Transform.TransformDerivatives
using Transform.ParamBox
using Transform.box_derivatives

blob, mp, body = gen_three_body_dataset();
transform = get_mp_transform(mp, loc_width=1.0);

transform_derivatives = TransformDerivatives{Float64}(length(mp.active_sources));

for param in fieldnames(ids), sa = 1:length(mp.active_sources)

	#println(param, " ", sa)
	constraint_vec = transform.bounds[sa][param]

	if isa(constraint_vec[1], ParamBox) # It is a box constraint
		@assert length(constraint_vec) == length(ids_free.(param)) == length(ids.(param))

		# Get each components' derivatives one by one.
		for ind = 1:length(constraint_vec)
			@assert isa(constraint_vec[ind], ParamBox)
			vp_ind = ids.(param)[ind]
			vp_free_ind = ids_free.(param)[ind]

			jac, hess = box_derivatives(mp.vp[s][vp_ind], constraint_vec[ind]);

			vp_sf_ind = length(CanonicalParams) * (sa - 1) + vp_ind
			vp_free_sf_ind = length(UnconstrainedParams) * (sa - 1) + vp_free_ind

			transform_derivatives.dparam_dfree[vp_sf_ind, vp_free_sf_ind] = jac
			transform_derivatives.d2param_dfree2[
				vp_sf_ind][vp_free_sf_ind, vp_free_sf_ind] = hess
		end
	else # It is a simplex constraint

			# If a param is not a box constraint, it must have all simplex constraints.
		@assert all([ isa(constraint, SimplexBox)  for constraint in constraint_vec])

		param_size = size(ids.(param))
		if length(param_size) == 2 # It's a simplex matrix
			@assert length(constraint_vec) == param_size[2]
			for col=1:(param_size[2])
				vp_free_ind = ids_free.(param)[:, col]
				vp_ind = ids.(param)[:, col]
				vp_sf_ind = length(CanonicalParams) * (sa - 1) + vp_ind
				vp_free_sf_ind = length(UnconstrainedParams) * (sa - 1) + vp_free_ind

				jac, hess = Transform.box_simplex_derivatives(
					mp.vp[s][vp_ind], constraint_vec[col])

				transform_derivatives.dparam_dfree[vp_sf_ind, vp_free_sf_ind] = jac
				for row in 1:(param_size[1])
					transform_derivatives.d2param_dfree2[
						vp_sf_ind[row]][vp_free_sf_ind, vp_free_sf_ind] = hess[row]
				end
			end
		else # It is simply a single simplex vector.
			@assert length(constraint_vec) == 1
			vp_free_ind = ids_free.(param)
			vp_ind = ids.(param)
			# Hack, see TODO in CelesteTypes.
			if length(free_ind) == 1
				vp_free_ind = Int64[ vp_free_ind ]
			end
			vp_sf_ind = length(CanonicalParams) * (sa - 1) + vp_ind
			vp_free_sf_ind = length(UnconstrainedParams) * (sa - 1) + vp_free_ind

			jac, hess = Transform.box_simplex_derivatives(
				mp.vp[s][vp_ind], constraint_vec[1])

			transform_derivatives.dparam_dfree[vp_sf_ind, vp_free_sf_ind] = jac
			for ind in 1:length(vp_ind)
				transform_derivatives.d2param_dfree2[
					vp_sf_ind[ind]][vp_free_sf_ind, vp_free_sf_ind] = hess[ind]
			end
		end
	end
end















println("Running transform tests.")


function test_box_derivatives()
	blob, mp, body = gen_three_body_dataset();
	transform = get_mp_transform(mp, loc_width=1.0);

	box_params = setdiff(fieldnames(ids), [:a, :k])
	vp_free = transform.from_vp(mp.vp)
	for sa = 1:length(mp.active_sources), param in box_params, ind in length(ids.(param))
		# sa = 1
		# param = box_params[1]
		# ind = 1

		s = mp.active_sources[sa]
		vp_ind = ids.(param)[ind]
		free_ind = [ids_free.(param)[ind]]

		function wrap_transform{NumType <: Number}(vp_free_s::Vector{NumType})
			local_vp_free =
				Array{NumType, 1}[ convert(Array{NumType, 1}, vp_free[sa]) for
			                  sa = 1:length(mp.active_sources) ]
			local_vp_free[s] = vp_free_s
			vp = transform.to_vp(local_vp_free)
			vp[s][ids.(param)[ind]]
		end

		ad_d  = ForwardDiff.gradient(wrap_transform, vp_free[s])[free_ind][1]
		ad_h = ForwardDiff.hessian(wrap_transform, vp_free[s])[free_ind, free_ind][1,1]

		d, h = Transform.box_derivatives(
			mp.vp[s][vp_ind][1], transform.bounds[s][param][ind])
		@test_approx_eq ad_d d
		@test_approx_eq ad_h h
	end
end


function test_box_simplex_derivatives()
	blob, mp, body = gen_three_body_dataset();
	for s = 1:mp.S
		delta = 0.01 * s # Make the parameters different for each one
		mp.vp[s][ids.a] = Float64[ 0.2 - delta, 0.8 + delta ]
		mp.vp[s][ids.k] = Float64[ 0.2- delta 0.2- delta; 0.8 + delta 0.8 + delta ]
	end
	transform = get_mp_transform(mp, loc_width=1.0);

	simplex_params = [:a, :k]
	vp_free = transform.from_vp(mp.vp)

	for sa = 1:length(mp.active_sources), param in simplex_params
		# sa = 1
		# param = :k
		# col = 1 # For k only
		# ind = 1 # Index within the simplex

		s = mp.active_sources[sa]
		num_cols = length(size(ids.(param)))
		@assert num_cols == 1 || num_cols == 2

		for col = 1:num_cols
			vp_ind = ids.(param)[:, col]

			if length(size(ids_free.(param))) == 0
				# Hack to handle ids_free.a
				@assert col == 1
				free_ind = [ ids_free.(param) ]
			else
				free_ind = ids_free.(param)[:, col]
			end

			d, h = Transform.box_simplex_derivatives(
				mp.vp[s][vp_ind], transform.bounds[s][param][col])

			for row = 1:2
				# Write with a univariate output so we can take autodiff hessians.
			  function wrap_transform{NumType <: Number}(vp_free_s::Vector{NumType})
			  	local_vp_free =
			  		Array{NumType, 1}[ convert(Array{NumType, 1}, vp_free[sa]) for
			  	                     sa = 1:length(mp.active_sources) ]
			  	local_vp_free[s] = vp_free_s
			  	vp = transform.to_vp(local_vp_free)
			  	vp[s][ids.(param)[row, col]]
			  end

			  ad_d = ForwardDiff.gradient(wrap_transform, vp_free[s])[free_ind]
			  ad_h = ForwardDiff.hessian(wrap_transform, vp_free[s])[free_ind, free_ind]
				@test_approx_eq ad_d d[row, :][1]
				@test_approx_eq ad_h h[row]
			end
		end
	end
end


function test_simplex_derivatives()
	n = 4
	basic_simplex_box = Transform.SimplexBox(0, 1, n)
	z = Float64[1, 2, 4, 3]
	z /= sum(z)
	r = Transform.unsimplexify_parameter(z, basic_simplex_box)
	Transform.simplexify_parameter(r, basic_simplex_box)

	ad_d = Array(Array{Float64}, n)
	ad_h = Array(Array{Float64}, n)

	for ind = 1:n
	  function wrap_simplex{NumType <: Number}(r::Vector{NumType})
	    local z = Transform.simplexify_parameter(r, basic_simplex_box)
	    z[ind]
	  end
	  ad_d[ind] = ForwardDiff.gradient(wrap_simplex, r)
	  ad_h[ind] = ForwardDiff.hessian(wrap_simplex, r)
	end

	jacobian, hessian_vec = Transform.simplex_derivatives(z)

	@test_approx_eq jacobian' reduce(hcat, ad_d)
	for ind = 1:n
		@test_approx_eq(hessian_vec[ind], ad_h[ind])
	end
end


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
		if NumType <: DualNumbers.Dual
			param = realpart(param)
			new_param = realpart(new_param)
		end
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
		if NumType <: DualNumbers.Dual
			param = realpart(param)
			new_param = realpart(new_param)
		end
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

test_transform_box_functions()


test_box_derivatives()
test_box_simplex_derivatives()
test_simplex_derivatives()
test_identity_transform()
test_parameter_conversion()
test_transform_simplex_functions()
test_basic_transforms()
