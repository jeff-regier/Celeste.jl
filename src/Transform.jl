# Convert between different parameterizations.

module Transform

using Celeste
using CelesteTypes
import DualNumbers

export DataTransform, ParamBounds, ParamBox
export get_mp_transform, generate_valid_parameters


################################
# Elementary functions.

@doc """
Unconstrain x in the unit interval to lie in R.
""" ->
function inv_logit{NumType <: Number}(x::NumType)
    @assert(x >= 0)
    @assert(x <= 1)
    -log(1.0 / x - 1)
end


function inv_logit{NumType <: Number}(x::Array{NumType})
    @assert(all(x .>= 0))
    @assert(all(x .<= 1))
    -log(1.0 ./ x - 1)
end


@doc """
Convert x in R to lie in the unit interval.
""" ->
function logit{NumType <: Number}(x::NumType)
    1.0 / (1.0 + exp(-x))
end


function logit{NumType <: Number}(x::Array{NumType})
    1.0 ./ (1.0 + exp(-x))
end


@doc """
Convert an (n - 1)-vector of real numbers to an n-vector on the simplex, where
the last entry implicitly has the untransformed value 1.
""" ->
function constrain_to_simplex{NumType <: Number}(x::Vector{NumType})
  if any(x .== Inf)
    z = NumType[ x_entry .== Inf ? one(NumType) : zero(NumType) for x_entry in x]
    z ./ sum(z)
    push!(z, 0)
    return(z)
  else
    z = exp(x)
    z_sum = sum(z) + 1
    z ./= z_sum
    push!(z, 1 / z_sum)
    return(z)
  end
end


@doc """
Convert an n-vector on the simplex to an (n - 1)-vector in R^{n -1}, where
the last entry implicitly has the untransformed value 1.
""" ->
function unconstrain_simplex{NumType <: Number}(z::Vector{NumType})
  n = length(z)
  NumType[ log(z[i]) - log(z[n]) for i = 1:(n - 1)]
end


################################
# The transforms for Celeste.

immutable ParamBox
  lower_bound::Float64
  upper_bound::Float64
  scale::Float64

  ParamBox(lower_bound, upper_bound, scale) = begin
    @assert lower_bound > -Inf # Not supported
    @assert scale > 0.0
    @assert lower_bound < upper_bound
    new(lower_bound, upper_bound, scale)
  end
end

immutable SimplexBox
  lower_bound::Float64
  scale::Float64
  n::Int64

  SimplexBox(lower_bound, scale, n) = begin
    @assert n >= 2
    @assert 0.0 <= lower_bound < 1 / n
    new(lower_bound, scale, n)
  end
end

# The box bounds for a symbol.  The tuple contains
# (lower bounds, upper bound, scale).
typealias ParamBounds Dict{Symbol, Union{Vector{ParamBox}, Vector{SimplexBox}}}


@doc """
Derivatives of the transform from free to constrained parameters.
""" ->
type TransformDerivatives{NumType <: Number}
  dparam_dfree::Matrix{NumType}
  d2param_dfree2::Array{Matrix{NumType}}

  # TODO: use sparse matrices?
  TransformDerivatives(S::Int64) = begin
    dparam_dfree =
      zeros(NumType,
            S * length(CanonicalParams), S * length(UnconstrainedParams))
    d2param_dfree2 = Array(Matrix{NumType}, S * length(CanonicalParams))
    for i in 1:(S * length(CanonicalParams))
      d2param_dfree2[i] =
        zeros(NumType,
              S * length(UnconstrainedParams), S * length(UnconstrainedParams))
    end
    new(dparam_dfree, d2param_dfree2)
  end
end

###############################################
# Functions for a "free transform".

function unbox_parameter{NumType <: Number}(param::NumType, param_box::ParamBox)

  lower_bound = param_box.lower_bound
  upper_bound = param_box.upper_bound
  scale = param_box.scale

  positive_constraint = (upper_bound == Inf)

  # exp and the logit functions handle infinities correctly, so
  # parameters can equal the bounds.
  if isa(param, DualNumbers.Dual)
    param_val = DualNumbers.realpart(param)
  else
    param_val = param
  end
  @assert(lower_bound .<= param_val .<= upper_bound,
          string("unbox_parameter: param outside bounds: ",
                 "$param ($lower_bound, $upper_bound)"))

  if positive_constraint
    return log(param - lower_bound) * scale
  else
    param_bounded = (param - lower_bound) / (upper_bound - lower_bound)
    return inv_logit(param_bounded) * scale
  end
end


function box_parameter{NumType <: Number}(
    free_param::NumType, param_box::ParamBox)

  lower_bound = param_box.lower_bound
  upper_bound = param_box.upper_bound
  scale = param_box.scale

  positive_constraint = (upper_bound == Inf)
  if positive_constraint
    return (exp(free_param / scale) + lower_bound)
  else
    return(
      logit(free_param / scale) * (upper_bound - lower_bound) + lower_bound)
  end
end


@doc """
Convert an unconstrained (n-1)-vector to a simplicial n-vector, z, such that
  - sum(z) = 1
  - z >= simplex_box.lower_bound
See notes for a derivation and reasoning.
""" ->
function simplexify_parameter{NumType <: Number}(
    free_param::Vector{NumType}, simplex_box::SimplexBox)

  n = simplex_box.n
  lower_bound = simplex_box.lower_bound
  scale = simplex_box.scale

  @assert length(free_param) == (n - 1)

  # Broadcasting doesn't work with DualNumbers and Floats. :(
  # z_sim is on an unconstrained simplex.
  z_sim = constrain_to_simplex(NumType[ p / scale for p in free_param ])
  param = NumType[ (1 - n * lower_bound) * p + lower_bound for p in z_sim ]

  param
end


@doc """
Invert the transformation simplexify_parameter()
""" ->
function unsimplexify_parameter{NumType <: Number}(
    param::Vector{NumType}, simplex_box::SimplexBox)

  n = simplex_box.n
  lower_bound = simplex_box.lower_bound
  scale = simplex_box.scale

  @assert length(param) == n
  @assert all(param .>= lower_bound)
  @assert abs(sum(param) - 1) < 1e-16

  # z_sim is on an unconstrained simplex.
  # Broadcasting doesn't work with DualNumbers and Floats. :(
  z_sim = NumType[ (p - lower_bound) / (1 - n * lower_bound) for p in param ]
  free_param = NumType[ p * scale for p in unconstrain_simplex(z_sim) ]

  free_param
end


##################
# Derivatives

@doc """
Return derivatives of an unscaled transform from free parameters to a simplex.

Args:
  - z_sim: A vector in a simplex (NB: the function returns derivatives of the
           function f(unconstrained) = simplex, but it the answer is expressed
           in terms of the output of the function.)

Returns:
  - jacobian: n by (n -1) matrix of derivatives of the simplex output (in rows)
              wrt the free parameters (in columns)
  - hessian_vec: An n-length vector of the hessian of each simplex output
                 parameter with respect to the (n-1) free input parameters.
""" ->
function simplex_derivatives{NumType <: Number}(z_sim::Vector{NumType})
	n = length(z_sim)
	hessian_vec = Array(Array{NumType}, n)
  for i = 1:n
    hessian_vec[i] = Array(NumType, n - 1, n - 1)
    for j=1:(n - 1), k=1:(n - 1)
      if j != k
        if (j == i)
          hessian_vec[i][j, k] = -z_sim[i] * z_sim[k] * (1 - 2 * z_sim[i])
        elseif (k == i)
					hessian_vec[i][j, k] = -z_sim[i] * z_sim[j] * (1 - 2 * z_sim[i])
        else
          hessian_vec[i][j, k] = 2 * z_sim[i] * z_sim[j] * z_sim[k]
        end
      else # j == k
        if i == j # All equal
          hessian_vec[i][j, k] = z_sim[i] * (1 - z_sim[j]) * (1 - 2 * z_sim[k])
        else # j == k, but both are different from i
          hessian_vec[i][j, k] = - z_sim[i] * z_sim[j] * (1 - 2 * z_sim[j])
        end
      end
    end
  end

	jacobian =
		NumType[ z_sim[i] * (i == j) - z_sim[i] * z_sim[j] for i=1:n, j=1:(n - 1) ]

	jacobian, hessian_vec
end


@doc """
Return the derivative and hessian of a simplex transform given the constrained
parameters.

Args:
  - param: The constrained parameter (NB: the derivatives are expressed
		       as a function of the constrained parameterd despite being
					 the derivative of the function unconstrained -> constrained)
	- simplex_box: A box simplex constraint
""" ->
function box_simplex_derivatives{NumType <: Number}(
    param::Vector{NumType}, simplex_box::SimplexBox)
	lower_bound = simplex_box.lower_bound
  scale = simplex_box.scale
  n = simplex_box.n

  @assert length(param) == n

  # z_sim is on an unconstrained simplex.
  # Broadcasting doesn't work with DualNumbers and Floats. :(
  z_sim = NumType[ (p - lower_bound) / (1 - n * lower_bound) for p in param ]

	jacobian, hessian_vec = simplex_derivatives(z_sim)

	for i in 1:n
		hessian_vec[i] *= (scale ^ 2) * (1 - n * lower_bound)
	end
	jacobian *= scale * (1 - n * lower_bound)
  jacobian, hessian_vec
end


@doc """
Return the derivative of a function that turns a free parameter into a
box-constrained parameter.

Args:
 - param: The value of the paramter that lies within the box constraints.
 - lower_bound: The lower bound of the box
 - upper_bound: The upper bound of the box
 - scale: The rescaling parameter of the unconstrained variable.

Returns:
  d(constrained parameter) / d (free parameter)
""" ->
function unbox_derivative{NumType <: Number}(
  param::Union{NumType, Array{NumType}},
  lower_bound::Union{Float64, Array{Float64}},
  upper_bound::Union{Float64, Array{Float64}},
  scale::Union{Float64, Array{Float64}})

    positive_constraint = any(upper_bound .== Inf)
    if positive_constraint && !all(upper_bound .== Inf)
      error(string("unbox_derivative: Some but not all upper bounds are Inf: ",
                   "$upper_bound"))
    end

    # Strict inequality is not required for derivatives.
    @assert(all(lower_bound .<= real(param) .<= upper_bound),
            string("unbox_derivative: param outside bounds: ",
                   "$param ($lower_bound, $upper_bound)"))

    if positive_constraint
      return (param - lower_bound) ./ scale
    else
      # Box constraints.
      param_scaled = (param - lower_bound) ./ (upper_bound - lower_bound)
      return (param_scaled .*
              (1 - param_scaled) .* (upper_bound - lower_bound) ./ scale)
    end
end


@doc """
Return the derivative of a function that turns a free parameter into a
simplex-constrained parameter.

Args:
 - param: The value of the paramter that lies within the simplex.
 - lower_bound: The lower bound of the simplex (not necessarily zero)
 - upper_bound: The upper bound of the simplex (not necessarily one)
 - scale: The rescaling parameter of the unconstrained variable.

Returns:
  d(constrained parameter) / d (free parameter)
""" ->
function inverse_simplex_derivative{NumType <: Number}(
  param::Union{NumType, Array{NumType}},
  lower_bound::Union{Float64, Array{Float64}},
  upper_bound::Union{Float64, Array{Float64}},
  scale::Union{Float64, Array{Float64}})

    positive_constraint = any(upper_bound .== Inf)
    if positive_constraint && !all(upper_bound .== Inf)
      error(string("unbox_derivative: Some but not all upper bounds are Inf: ",
                   "$upper_bound"))
    end

    # Strict inequality is not required for derivatives.
    @assert(all(lower_bound .<= real(param) .<= upper_bound),
            string("unbox_derivative: param outside bounds: ",
                   "$param ($lower_bound, $upper_bound)"))

    if positive_constraint
      return (param - lower_bound) ./ scale
    else
      # Box constraints.
      param_scaled = (param - lower_bound) ./ (upper_bound - lower_bound)
      return (param_scaled .*
              (1 - param_scaled) .* (upper_bound - lower_bound) ./ scale)
    end
end


@doc """
Return the derivative and hessian of a box transform given the constrained
parameters.

Args:
  - param: The constrained parameter (NB: the derivatives are expressed
		       as a function of the constrained parameterd despite being
					 the derivative of the function unconstrained -> constrained)
	- param_box: A box constraint
""" ->
function box_derivatives{NumType <: Number}(param::NumType, param_box::ParamBox)
	lower_bound = param_box.lower_bound
  upper_bound = param_box.upper_bound
  scale = param_box.scale

	if upper_bound == Inf
		centered_param = param - lower_bound
		return scale * centered_param, scale ^ 2 * centered_param
	else
		param_range = upper_bound - lower_bound
		centered_param = (param - lower_bound) / param_range
		derivative = param_range * centered_param * (1 - centered_param) / scale
		return derivative, derivative * (1 - 2 * centered_param) / scale
	end
end


######################
# Functions to take actual parameter vectors.

@doc """
Convert a variational parameter vector to an unconstrained version.
""" ->
function perform_transform!{NumType <: Number}(
    vp::Vector{NumType}, vp_free::Vector{NumType}, bounds::ParamBounds,
    to_unconstrained::Bool)

  for (param, constraint_vec) in bounds

    is_box = isa(bounds[param], Array{ParamBox})
    if is_box
      # Apply a box constraint to each parameter.
      @assert(length(ids.(param)) == length(ids_free.(param)) ==
        length(bounds[param]))
      for ind in 1:length(ids.(param))
        constraint = constraint_vec[ind]
        free_ind = ids_free.(param)[ind]
        vp_ind = ids.(param)[ind]
        to_unconstrained ?
          vp_free[free_ind] = unbox_parameter(vp[vp_ind], constraint):
          vp[vp_ind] = box_parameter(vp_free[free_ind], constraint)
      end
    else
      # Apply a simplex constraint to each parameter.
      @assert isa(bounds[param], Array{SimplexBox})
      # Some simplicial parameters are vectors, some are matrices.  Which is
      # which is determined by the size of the ids.
      param_size = size(ids.(param))
      if length(param_size) == 2
        # If the ids are a matrix, then each column is a simplex.
        # Each column should have its own simplicial constrains.
        @assert length(bounds[param]) == param_size[2]
        for col in 1:(param_size[2])
          free_ind = ids_free.(param)[:, col]
          vp_ind = ids.(param)[:, col]
          constraint = constraint_vec[col]
          to_unconstrained ?
            vp_free[free_ind] = unsimplexify_parameter(vp[vp_ind], constraint):
            vp[vp_ind] = simplexify_parameter(vp_free[free_ind], constraint)
        end
      else
        # It is simply a simplex vector.
        @assert length(bounds[param]) == 1
        free_ind = ids_free.(param)
        vp_ind = ids.(param)
        # Hack, see TODO in CelesteTypes.
        if length(free_ind) == 1
          free_ind = Int64[ free_ind ]
        end
        constraint = constraint_vec[1]
        to_unconstrained ?
          vp_free[free_ind] = unsimplexify_parameter(vp[vp_ind], constraint):
          vp[vp_ind] = simplexify_parameter(vp_free[free_ind], constraint)
      end
    end
  end
end


function free_to_vp!{NumType <: Number}(
    vp_free::Vector{NumType}, vp::Vector{NumType}, bounds::ParamBounds)

  perform_transform!(vp, vp_free, bounds, false)
end


function vp_to_free!{NumType <: Number}(
    vp::Vector{NumType}, vp_free::Vector{NumType}, bounds::ParamBounds)

  perform_transform!(vp, vp_free, bounds, true)
end


#####################
# Conversion to and from variational parameter vectors and arrays.

@doc """
Transform VariationalParams to an array.

vp = variational parameters
omitted_ids = ids in ParamIndex

There is probably no use for this function, since you'll only be passing
trasformations to the optimizer, but I'll include it for completeness.""" ->
function free_vp_to_array{NumType <: Number}(vp::FreeVariationalParams{NumType},
                                             omitted_ids::Vector{Int64})

    left_ids = setdiff(1:length(UnconstrainedParams), omitted_ids)
    new_P = length(left_ids)
    S = length(vp)
    x_new = zeros(NumType, new_P, S)

    for p1 in 1:length(left_ids), s=1:S
        p0 = left_ids[p1]
        x_new[p1, s] = vp[s][p0]
    end

    x_new
end

@doc """
Transform a parameter vector to variational parameters in place.

Args:
 - xs: A (param x sources) matrix created from free variational parameters.
 - vp_free: Free variational parameters.  Only the ids not in omitted_ids
            will be updated.
 - omitted_ids: Ids to omit (from ids_free)

Returns:
 - Update vp_free in place.
""" ->
function array_to_free_vp!{NumType <: Number}(
    xs::Matrix{NumType}, vp_free::FreeVariationalParams{NumType},
    omitted_ids::Vector{Int64})

    left_ids = setdiff(1:length(UnconstrainedParams), omitted_ids)
    P = length(left_ids)
    S = length(vp_free)
    @assert size(xs) == (P, S)

    for s in 1:S, p1 in 1:P
        p0 = left_ids[p1]
        vp_free[s][p0] = xs[p1, s]
    end
end


@doc """
Generate parameters within the given bounds.
""" ->
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
      for ind in 1:length(ids.(param))
        constraint = constraint_vec[ind]
        constraint.upper_bound == Inf ?
          vp[s][ids.(param)[ind]] = constraint.lower_bound + 1.0:
          vp[s][ids.(param)[ind]] =
            0.5 * (constraint.upper_bound - constraint.lower_bound) +
            constraint.lower_bound
      end
    else
      # Simplex parameters can ignore the bounds.
      param_size = size(ids.(param))
      if length(param_size) == 2
        # matrix simplex
        for col in 1:param_size[2]
          vp[s][ids.(param)[:, col]] = 1 / param_size[1]
        end
      else
        # vector simplex
        vp[s][ids.(param)] = 1 / length(ids.(param))
      end
    end
	end

  vp
end


#########################
# Define the exported variables.

@doc """
Functions to move between a single source's variational parameters and a
transformation of the data for optimization.

to_vp: A function that takes transformed parameters and returns
       variational parameters
from_vp: A function that takes variational parameters and returned
         transformed parameters
to_vp!: A function that takes (transformed paramters, variational parameters)
        and updates the variational parameters in place
from_vp!: A function that takes (variational paramters, transformed parameters)
          and updates the transformed parameters in place
...
transform_sensitive_float: A function that takes (sensitive float, model
  parameters) where the sensitive float contains partial derivatives with
  respect to the variational parameters and returns a sensitive float with total
  derivatives with respect to the transformed parameters.
bounds: The bounds for each parameter and each object in ModelParams.
active_sources: The sources that are being optimized.  Only these sources'
  parameters are transformed into the parameter vector.
  """ ->
type DataTransform
	to_vp::Function
	from_vp::Function
	to_vp!::Function
	from_vp!::Function
  vp_to_array::Function
  array_to_vp!::Function
	transform_sensitive_float::Function
  bounds::Vector{ParamBounds}
  active_sources::Vector{Int64}
  active_S::Int64
  S::Int64
end

# TODO: Maybe this should be initialized with ModelParams with optional
# custom bounds.  Or maybe it should be part of ModelParams with one transform
# per celestial object rather than a single object containing an array of
# transforms.
DataTransform(bounds::Vector{ParamBounds};
              active_sources=collect(1:length(bounds)), S=length(bounds)) = begin

  @assert length(bounds) == length(active_sources)
  @assert maximum(active_sources) <= S
  active_S = length(active_sources)

  # Make sure that each variable has its bounds set.  The simplicial variables
  # :a and :k don't have bounds.
  for s=1:length(bounds)
    @assert Set(keys(bounds[s])) == Set(fieldnames(ids))
  end

  function from_vp!{NumType <: Number}(
    vp::VariationalParams{NumType}, vp_free::VariationalParams{NumType})
      S = length(vp)
      @assert length(vp_free) == active_S
      for si=1:active_S
        s = active_sources[si]
        vp_to_free!(vp[s], vp_free[si], bounds[si])
      end
  end

  function from_vp{NumType <: Number}(vp::VariationalParams{NumType})
      vp_free = [ zeros(NumType, length(ids_free)) for si = 1:active_S]
      from_vp!(vp, vp_free)
      vp_free
  end

  function to_vp!{NumType <: Number}(
    vp_free::FreeVariationalParams{NumType}, vp::VariationalParams{NumType})
      @assert length(vp_free) == active_S
      for si=1:active_S
        s = active_sources[si]
        free_to_vp!(vp_free[si], vp[s], bounds[si])
      end
  end

  function to_vp{NumType <: Number}(vp_free::FreeVariationalParams{NumType})
      @assert(active_S == S,
              string("to_vp is not supported when active_sources is a ",
                     "strict subset of all sources."))
      vp = [ zeros(NumType, length(CanonicalParams)) for s = 1:S]
      to_vp!(vp_free, vp)
      vp
  end

  function vp_to_array{NumType <: Number}(vp::VariationalParams{NumType},
                                          omitted_ids::Vector{Int64})
      vp_trans = from_vp(vp)
      free_vp_to_array(vp_trans, omitted_ids)
  end

  function array_to_vp!{NumType <: Number}(xs::Matrix{NumType},
                                           vp::VariationalParams{NumType},
                                           omitted_ids::Vector{Int64})
      # This needs to update vp in place so that variables in omitted_ids
      # stay at their original values.
      vp_trans = from_vp(vp)
      array_to_free_vp!(xs, vp_trans, omitted_ids)
      to_vp!(vp_trans, vp)
  end

  # Given a sensitive float with derivatives with respect to all the
  # constrained parameters, calculate derivatives with respect to
  # the unconstrained parameters.
  #
  # Note that all the other functions in ElboDeriv calculated derivatives with
  # respect to the constrained parameterization.
  function transform_sensitive_float{NumType <: Number}(
    sf::SensitiveFloat, mp::ModelParams{NumType})

      # Require that the input have all derivatives defined, even for the
      # non-active sources.
      @assert size(sf.d) == (length(CanonicalParams), S)
      @assert mp.S == S

      sf_free = zero_sensitive_float(UnconstrainedParams, NumType, active_S)
      sf_free.v = sf.v

      function unbox_wrapper{NumType <: Number}(vp_vec::Vector{NumType})
        vp = reshape()
      end

      for si in 1:active_S
        s = active_sources[si]
        sf_free.d[:, si] =
          unbox_param_derivative(mp.vp[s], sf.d[:, s][:], bounds[si])
      end

      sf_free
  end

  DataTransform(to_vp, from_vp, to_vp!, from_vp!, vp_to_array, array_to_vp!,
                transform_sensitive_float, bounds, active_sources, active_S, S)
end

function get_mp_transform(mp::ModelParams; loc_width::Float64=1.5e-3)
  bounds = Array(ParamBounds, length(mp.active_sources))

  # Note that, for numerical reasons, the bounds must be on the scale
  # of reasonably meaningful changes.
  for si in 1:length(mp.active_sources)
    s = mp.active_sources[si]
    bounds[si] = ParamBounds()
    bounds[si][:u] = Array(ParamBox, 2)
    u = mp.vp[s][ids.u]
    for axis in 1:2
      bounds[si][:u][axis] =
        ParamBox(u[axis] - loc_width, u[axis] + loc_width, 1.0)
    end
    bounds[si][:r1] = Array(ParamBox, Ia)
    bounds[si][:r2] = Array(ParamBox, Ia)
    for i in 1:Ia
      bounds[si][:r1][i] = ParamBox(-1.0, 10., 1.0)
      bounds[si][:r2][i] = ParamBox(1e-4, 0.1, 1.0)
    end
    bounds[si][:c1] = Array(ParamBox, 4 * Ia)
    bounds[si][:c2] = Array(ParamBox, 4 * Ia)
    for ind in 1:length(ids.c1)
      bounds[si][:c1][ind] = ParamBox(-10., 10., 1.0)
      bounds[si][:c2][ind] = ParamBox(1e-4, 1., 1.0)
    end
    bounds[si][:e_dev] = ParamBox[ ParamBox(1e-2, 1 - 1e-2, 1.0) ]
    bounds[si][:e_axis] = ParamBox[ ParamBox(1e-2, 1 - 1e-2, 1.0) ]
    bounds[si][:e_angle] = ParamBox[ ParamBox(-10.0, 10.0, 1.0) ]
    bounds[si][:e_scale] = ParamBox[ ParamBox(0.1, 70., 1.0) ]

    const simplex_min = 0.005
    bounds[si][:a] = SimplexBox[ SimplexBox(simplex_min, 1.0, 2) ]

    bounds[si][:k] = Array(SimplexBox, D)
    for d in 1:D
      bounds[si][:k][d] = SimplexBox(simplex_min, 1.0, 2)
    end
  end
  DataTransform(bounds, active_sources=mp.active_sources, S=mp.S)
end

# An identity transform that does not enforce any bounds nor reduce
# the dimension of the variational params.  This is mostly useful
# for testing.
function get_identity_transform(P::Int64, S::Int64)

  active_S = S
  active_sources = 1:S
  bounds = ParamBounds[]

  function from_vp!{NumType <: Number}(
    vp::VariationalParams{NumType}, vp_free::VariationalParams{NumType})
      @assert length(vp_free) == length(vp) == active_S
      for s=1:S
        @assert length(vp_free[s]) == length(vp[s]) == P
        vp_free[s][:] = vp[s]
      end
  end

  function from_vp{NumType <: Number}(vp::VariationalParams{NumType})
      vp_free = [ zeros(NumType, P) for si = 1:active_S]
      from_vp!(vp, vp_free)
      vp_free
  end

  function to_vp!{NumType <: Number}(
    vp_free::FreeVariationalParams{NumType}, vp::VariationalParams{NumType})
    @assert length(vp_free) == length(vp) == active_S
    for s=1:S
      @assert length(vp_free[s]) == length(vp[s]) == P
      vp[s][:] = vp_free[s]
    end
  end

  function to_vp{NumType <: Number}(vp_free::FreeVariationalParams{NumType})
      vp = [ zeros(P) for s = 1:S]
      to_vp!(vp_free, vp)
      vp
  end

  function vp_to_array{NumType <: Number}(vp::VariationalParams{NumType},
                                          omitted_ids::Vector{Int64})
      kept_ids = setdiff(1:P, omitted_ids)
      xs = zeros(length(kept_ids), S)
      for s=1:S
        xs[:, s] = vp[s][kept_ids]
      end
      xs
  end

  function array_to_vp!{NumType <: Number}(xs::Matrix{NumType},
                                           vp::VariationalParams{NumType},
                                           omitted_ids::Vector{Int64})
      # This needs to update vp in place so that variables in omitted_ids
      # stay at their original values.
      kept_ids = setdiff(1:P, omitted_ids)
      for s=1:S
        vp[s][kept_ids] = xs[:, s]
      end
  end

  function transform_sensitive_float{NumType <: Number}(
      sf::SensitiveFloat, mp::ModelParams{NumType})
    sf
  end

  DataTransform(to_vp, from_vp, to_vp!, from_vp!, vp_to_array, array_to_vp!,
                transform_sensitive_float, bounds, active_sources, active_S, S)
end



end
