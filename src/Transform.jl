# Convert between different parameterizations.

module Transform

using Compat

# TODO: don't import Model; transformations should operate on
# generic ParamSets
using ..SensitiveFloats
import ..Log

import ..Model: NUM_SOURCE_TYPES, ParamSet, NUM_BANDS, NUM_COLOR_COMPONENTS

export DataTransform, ParamBounds, ParamBox, SimplexBox,
       get_mp_transform, enforce_bounds!,
       VariationalParams, FreeVariationalParams


# A vector of variational parameters.  The outer index is
# of celestial objects, and the inner index is over individual
# parameters for that object (referenced using ParamIndex).
const VariationalParams{T <: Number} = Vector{Vector{T}}
const FreeVariationalParams{T <: Number} = Vector{Vector{T}}

#####################################################################################
# this is essentially a compatibility layer since Model has gotten rid of this code
# it's messy, but this doesn't really matter too much, since Transforms.jl is going
# the way of the dinosaur

struct CanonicalParams <: ParamSet
    pos::Vector{Int}
    gal_frac_dev::Int
    gal_axis_ratio::Int
    gal_angle::Int
    gal_radius_px::Int
    flux_loc::Vector{Int}
    flux_scale::Vector{Int}
    color_mean::Matrix{Int}
    color_var::Matrix{Int}
    a::Matrix{Int}
    k::Matrix{Int}
    CanonicalParams() =
        new([1, 2], 3, 4, 5, 6,
            collect(7:(7+NUM_SOURCE_TYPES-1)),  # flux_loc
            collect((7+NUM_SOURCE_TYPES):(7+2NUM_SOURCE_TYPES-1)), # flux_scale
            reshape((7+2NUM_SOURCE_TYPES):(7+2NUM_SOURCE_TYPES+(NUM_BANDS-1)*NUM_SOURCE_TYPES-1), (NUM_BANDS-1, NUM_SOURCE_TYPES)),  # color_mean
            reshape((7+2NUM_SOURCE_TYPES+(NUM_BANDS-1)*NUM_SOURCE_TYPES):(7+2NUM_SOURCE_TYPES+2*(NUM_BANDS-1)*NUM_SOURCE_TYPES-1), (NUM_BANDS-1, NUM_SOURCE_TYPES)),  # color_var
            reshape((7+2NUM_SOURCE_TYPES+2*(NUM_BANDS-1)*NUM_SOURCE_TYPES):(7+3NUM_SOURCE_TYPES+2*(NUM_BANDS-1)*NUM_SOURCE_TYPES-1), (NUM_SOURCE_TYPES, 1)),  # a
            reshape((7+3NUM_SOURCE_TYPES+2*(NUM_BANDS-1)*NUM_SOURCE_TYPES):(7+3NUM_SOURCE_TYPES+2*(NUM_BANDS-1)*NUM_SOURCE_TYPES+NUM_COLOR_COMPONENTS*NUM_SOURCE_TYPES-1), (NUM_COLOR_COMPONENTS, NUM_SOURCE_TYPES))) # k
end
const ids = CanonicalParams()
Base.length(::Type{CanonicalParams}) = 6 + 3*NUM_SOURCE_TYPES + 2*(NUM_BANDS-1)*NUM_SOURCE_TYPES + NUM_COLOR_COMPONENTS*NUM_SOURCE_TYPES

struct UnconstrainedParams <: ParamSet
    pos::Vector{Int}
    gal_frac_dev::Int
    gal_axis_ratio::Int
    gal_angle::Int
    gal_radius_px::Int
    flux_loc::Vector{Int}
    flux_scale::Vector{Int}
    color_mean::Matrix{Int}
    color_var::Matrix{Int}
    a::Matrix{Int}
    k::Matrix{Int}
    UnconstrainedParams() =
        new([1, 2], 3, 4, 5, 6,
            collect(7:(7+NUM_SOURCE_TYPES-1)),  # flux_loc
            collect((7+NUM_SOURCE_TYPES):(7+2NUM_SOURCE_TYPES-1)), # flux_scale
            reshape((7+2NUM_SOURCE_TYPES):(7+2NUM_SOURCE_TYPES+(NUM_BANDS-1)*NUM_SOURCE_TYPES-1), (NUM_BANDS-1, NUM_SOURCE_TYPES)),  # color_mean
            reshape((7+2NUM_SOURCE_TYPES+(NUM_BANDS-1)*NUM_SOURCE_TYPES):(7+2NUM_SOURCE_TYPES+2*(NUM_BANDS-1)*NUM_SOURCE_TYPES-1), (NUM_BANDS-1, NUM_SOURCE_TYPES)),  # color_var
            reshape((7+2NUM_SOURCE_TYPES+2*(NUM_BANDS-1)*NUM_SOURCE_TYPES):
                    (7+2NUM_SOURCE_TYPES+2*(NUM_BANDS-1)*NUM_SOURCE_TYPES+(NUM_SOURCE_TYPES-1)-1), (NUM_SOURCE_TYPES - 1, 1)),  # a
            reshape((7+2NUM_SOURCE_TYPES+2*(NUM_BANDS-1)*NUM_SOURCE_TYPES+(NUM_SOURCE_TYPES-1)):
                    (7+2NUM_SOURCE_TYPES+2*(NUM_BANDS-1)*NUM_SOURCE_TYPES+(NUM_SOURCE_TYPES-1)+(NUM_COLOR_COMPONENTS-1)*NUM_SOURCE_TYPES-1), (NUM_COLOR_COMPONENTS-1, NUM_SOURCE_TYPES))) # k
end
const ids_free = UnconstrainedParams()
Base.length(::Type{UnconstrainedParams}) =  6 + 2*NUM_SOURCE_TYPES + 2*(NUM_BANDS-1)*NUM_SOURCE_TYPES + (NUM_COLOR_COMPONENTS-1)*NUM_SOURCE_TYPES + NUM_SOURCE_TYPES-1

################################
# Elementary functions.

"""
Unconstrain x in the unit interval to lie in R.
"""
function inv_logit(x::Number)
    @assert(x >= 0)
    @assert(x <= 1)
    -log(1.0 / x - 1)
end


"""
Convert x in R to lie in the unit interval.
"""
function logit(x::Number)
    1.0 / (1.0 + exp(-x))
end


"""
Convert an (n - 1)-vector of real numbers to an n-vector on the simplex, where
the last entry implicitly has the untransformed value 1.
"""
function constrain_to_simplex(x::Vector{T}) where {T<:Number}
    m = maximum(x)
    if m == Inf
        # If more than 1 entry in x is Inf, it may be because the
        # the last entry in z is 0. Here we set all those entries to the
        # same value, though that may not be strictly correct.
        z   = T[ x_entry .== Inf ? one(T) : zero(T) for x_entry in x]
        z ./= sum(z)
        push!(z, 0)
        return z
    else
        z = exp.(x .- m)
        z_sum = sum(z) + exp(-m)
        z ./= z_sum
        push!(z, inv(z_sum)*exp(-m))
        return z
    end
end


"""
Convert an n-vector on the simplex to an (n - 1)-vector in R^{n -1}.  Entries
are expressed relative to the last element.
"""
function unconstrain_simplex(z::Vector{T}) where {T<:Number}
    n = length(z)
    T[ log(z[i]) - log(z[n]) for i = 1:(n - 1)]
end


################################
# The transforms for Celeste.

struct ParamBox
    lb::Float64  # lower bound
    ub::Float64  # upper bound
    scale::Float64

    function ParamBox(lb, ub, scale)
        @assert lb > -Inf # Not supported
        @assert scale > 0.0
        @assert lb < ub
        new(lb, ub, scale)
    end
end

struct SimplexBox
    lb::Float64  # lower bound
    scale::Float64
    n::Int

    function SimplexBox(lb, scale, n)
        @assert n >= 2
        @assert 0.0 <= lb < 1 / n
        new(lb, scale, n)
    end
end

# The vector of transform parameters for a Symbol.
const ParamBounds = Dict{Symbol, Union{Vector{ParamBox}, Vector{SimplexBox}}}


###############################################
# Functions for a "free transform".

function unbox_parameter(param::Number, pb::ParamBox)
    positive_constraint = (pb.ub == Inf)

    # exp and the logit functions handle infinities correctly, so
    # parameters can equal the bounds.
    @assert(pb.lb .<= param .<= pb.ub,
                    string("unbox_parameter: param outside bounds: ",
                                 "$param ($(pb.lb), $(pb.ub))"))

    if positive_constraint
        return log(param - pb.lb) * pb.scale
    else
        param_bounded = (param - pb.lb) / (pb.ub - pb.lb)
        return inv_logit(param_bounded) * pb.scale
    end
end


function box_parameter(free_param::Number, pb::ParamBox)
    positive_constraint = (pb.ub == Inf)
    if positive_constraint
        return exp(free_param / pb.scale) + pb.lb
    else
        return logit(free_param / pb.scale) * (pb.ub - pb.lb) + pb.lb
    end
end


"""
Convert an unconstrained (n-1)-vector to a simplicial n-vector, z, such that
  - sum(z) = 1
  - z >= sb.lb
See notes for a derivation and reasoning.
"""
function simplexify_parameter(free_param::Vector{T},
                              sb::SimplexBox) where {T<:Number}
    @assert length(free_param) == (sb.n - 1)

    # Broadcasting doesn't work with DualNumbers and Floats. :(
    # z_sim is on an unconstrained simplex.
    z_sim = constrain_to_simplex(T[ p / sb.scale for p in free_param ])
    param = T[ (1 - sb.n * sb.lb) * p + sb.lb for p in z_sim ]

    param
end


"""
Invert the transformation simplexify_parameter() by converting an n-vector
on a simplex to R^{n - 1}.
"""
function unsimplexify_parameter(param::Vector{T},
                                sb::SimplexBox) where {T<:Number}
    @assert length(param) == sb.n
    @assert all(param .>= sb.lb)
    @assert(abs(sum(param) - 1) < 1e-14, abs(sum(param) - 1))

    # z_sim is on an unconstrained simplex.
    # Broadcasting doesn't work with DualNumbers and Floats. :(
    z_sim = T[ (p - sb.lb) / (1 - sb.n * sb.lb) for p in param ]
    free_param = T[ p * sb.scale for p in unconstrain_simplex(z_sim) ]

    free_param
end


##################
# Derivatives

"""
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
"""
function simplex_derivatives(z_sim::Vector{T}) where {T<:Number}
    n = length(z_sim)
    hessian_vec = Vector{Array{T}}(n)

    for i = 1:n
        hessian_vec[i] = Matrix{T}(n - 1, n - 1)
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
        T[ z_sim[i] * (i == j) - z_sim[i] * z_sim[j] for i=1:n, j=1:(n - 1) ]

    jacobian, hessian_vec
end


"""
Return the derivative and hessian of a simplex transform given the constrained
parameters.

Args:
  - param: The constrained parameter (NB: the derivatives are expressed
               as a function of the constrained parameterd despite being
                     the derivative of the function unconstrained -> constrained)
    - sb: A box simplex constraint
"""
function box_simplex_derivatives(param::Vector{T},
                                 sb::SimplexBox) where {T<:Number}

    @assert length(param) == sb.n

    # z_sim is on an unconstrained simplex.
    # Broadcasting doesn't work with DualNumbers and Floats. :(
    z_sim = T[ (p - sb.lb) / (1 - sb.n * sb.lb) for p in param ]

    jacobian, hessian_vec = simplex_derivatives(z_sim)
    for i in 1:sb.n
        hessian_vec[i] *= (sb.scale ^ 2) * (1 - sb.n * sb.lb)
    end
    jacobian *= sb.scale * (1 - sb.n * sb.lb)

    jacobian, hessian_vec
end


"""
Return the derivative and hessian of a box transform given the constrained
parameters.

Args:
  - param: The constrained parameter (NB: the derivatives are expressed
               as a function of the constrained parameterd despite being
                     the derivative of the function unconstrained -> constrained)
    - pb: A box constraint
"""
function box_derivatives(param::Number, pb::ParamBox)
    if pb.ub == Inf
        centered_param = param - pb.lb
        return pb.scale * centered_param, pb.scale ^ 2 * centered_param
    else
        param_range = pb.ub - pb.lb
        centered_param = (param - pb.lb) / param_range
        derivative = param_range * centered_param * (1 - centered_param) / pb.scale
        return derivative, derivative * (1 - 2 * centered_param) / pb.scale
    end
end


"""
A datatype containing derivatives of the transform from free to constrained
parameters.

Members:
  dparam_dfree: The Jacobian of the transformation
                contrainted_param = f(free param)
  d2param_dfree2: A vector of hessians.  Each element is the Hessian of one
                  component of the aforementioned f()
"""
struct TransformDerivatives{T<:Number}
    dparam_dfree::Matrix{T}
    d2param_dfree2::Vector{Matrix{T}}
    Sa::Int

    # TODO: use sparse matrices?
    function TransformDerivatives{T}(Sa::Int) where {T<:Number}
        dparam_dfree =
            zeros(T, Sa * length(CanonicalParams), Sa * length(UnconstrainedParams))
        d2param_dfree2 = Vector{Matrix{T}}(Sa * length(CanonicalParams))
        for i in 1:(Sa * length(CanonicalParams))
            d2param_dfree2[i] =
                zeros(T, Sa * length(UnconstrainedParams), Sa * length(UnconstrainedParams))
        end
        new(dparam_dfree, d2param_dfree2, Sa)
    end
end


"""
Populate a TransformDerivatives object in place.

Args:
  - bounds: A vector containing one ParamBounds for each active source in the
            same order as active_sources.
  - transform_derivatives: TransformDerivatives to be populated.

Returns:
  Update transform_derivatives in place.
"""
function get_transform_derivatives!(
        vp::VariationalParams,
        active_sources::Vector{Int},
        bounds::Vector{ParamBounds},
        transform_derivatives::TransformDerivatives)

    @assert transform_derivatives.Sa == length(active_sources)

    for param in fieldnames(ids), sa = 1:length(active_sources)
        s = active_sources[sa]
        constraint_vec = bounds[sa][param]

        if isa(constraint_vec[1], ParamBox) # It is a box constraint
            @assert(length(constraint_vec) ==
                    length(getfield(ids_free, param)) ==
                    length(getfield(ids, param)))

            # Get each components' derivatives one by one.
            for ind = 1:length(constraint_vec)
                @assert isa(constraint_vec[ind], ParamBox)
                vp_ind = getfield(ids, param)[ind]
                vp_free_ind = getfield(ids_free, param)[ind]

                jac, hess = box_derivatives(vp[s][vp_ind], constraint_vec[ind])

                vp_sf_ind = length(CanonicalParams) * (sa - 1) + vp_ind
                vp_free_sf_ind = length(UnconstrainedParams) * (sa - 1) + vp_free_ind

                transform_derivatives.dparam_dfree[vp_sf_ind, vp_free_sf_ind] = jac
                transform_derivatives.d2param_dfree2[vp_sf_ind][vp_free_sf_ind, vp_free_sf_ind] = hess
            end
        else # It is a simplex constraint

            # If a param is not a box constraint, it must have all simplex constraints.
            @assert all([ isa(constraint, SimplexBox)  for constraint in constraint_vec])

            param_size = size(getfield(ids, param))
            @assert length(constraint_vec) == param_size[2]
            for col=1:(param_size[2])
                vp_free_ind = getfield(ids_free, param)[:, col]
                vp_ind = getfield(ids, param)[:, col]
                vp_sf_ind = length(CanonicalParams) * (sa - 1) + vp_ind
                vp_free_sf_ind = length(UnconstrainedParams) * (sa - 1) + vp_free_ind

                jac, hess = Transform.box_simplex_derivatives(
                    vp[s][vp_ind], constraint_vec[col])

                transform_derivatives.dparam_dfree[vp_sf_ind, vp_free_sf_ind] = jac
                for row in 1:(param_size[1])
                    transform_derivatives.d2param_dfree2[
                         vp_sf_ind[row]][vp_free_sf_ind, vp_free_sf_ind] = hess[row]
                end
            end
        end
    end
end


function get_transform_derivatives(vp::VariationalParams,
                                   active_sources::Vector{Int},
                                   bounds::Vector{ParamBounds})
  transform_derivatives =
    TransformDerivatives{Float64}(length(active_sources))
  get_transform_derivatives!(vp, active_sources, bounds, transform_derivatives)
  transform_derivatives
end


######################
# Functions to take actual parameter vectors.

"""
Convert between variational parameter vectors and unconstrained parameters.

Args:
  - vp: The vector of contrainted variationl parameters
  - vp_free: The vector of uncontrainted parameters for optimization
  - bounds: ParamBounds describing the transformation
  - to_unconstrained: If true, converts vp to vp_free.  If false, converts
                      vp_free to vp.

Returns:
  If to_unconstrained is true, updates vp_free in place.
  If to_unconstrained is false, updates vp in place.
"""
function perform_transform!(vp::Vector{T},
                            vp_free::Vector{T},
                            bounds::ParamBounds,
                            to_unconstrained::Bool) where {T<:Number}
    for (param, constraint_vec) in bounds
        is_box = isa(bounds[param], Array{ParamBox})
        if is_box
            # Apply a box constraint to each parameter.
            @assert(length(getfield(ids, param)) == length(getfield(ids_free, param)) ==
                length(bounds[param]))
            for ind in 1:length(getfield(ids, param))
                constraint = constraint_vec[ind]
                free_ind = getfield(ids_free, param)[ind]
                vp_ind = getfield(ids, param)[ind]
                to_unconstrained ?
                    vp_free[free_ind] = unbox_parameter(vp[vp_ind], constraint):
                    vp[vp_ind] = box_parameter(vp_free[free_ind], constraint)
            end
        else
            # Apply a simplex constraint to each parameter.
            @assert isa(bounds[param], Array{SimplexBox})
            param_size = size(getfield(ids, param))

            # Each column is a simplex and should have its own
            # simplicial constraints.
            @assert length(bounds[param]) == param_size[2]
            for col in 1:(param_size[2])
                free_ind = getfield(ids_free, param)[:, col]
                vp_ind = getfield(ids, param)[:, col]
                constraint = constraint_vec[col]
                to_unconstrained ?
                    vp_free[free_ind] = unsimplexify_parameter(vp[vp_ind], constraint):
                    vp[vp_ind] = simplexify_parameter(vp_free[free_ind], constraint)
            end
        end
    end
end


function free_to_vp!(vp_free::Vector{T},
                     vp::Vector{T},
                     bounds::ParamBounds) where {T<:Number}
    perform_transform!(vp, vp_free, bounds, false)
end


function vp_to_free!(vp::Vector{T},
                     vp_free::Vector{T},
                     bounds::ParamBounds) where {T<:Number}
    perform_transform!(vp, vp_free, bounds, true)
end


#####################
# Conversion to and from variational parameter vectors and arrays.

"""
Transform VariationalParams to an array.

Args:
  - vp = variational parameters
  - omitted_ids = ids in ParamIndex

There is probably no use for this function, since you'll only be passing
trasformations to the optimizer, but I'll include it for completeness.
"""
function free_vp_to_array(vp::FreeVariationalParams{T},
                          omitted_ids::Vector{Int}) where {T<:Number}

    left_ids = setdiff(1:length(UnconstrainedParams), omitted_ids)
    new_P = length(left_ids)
    S = length(vp)
    x_new = zeros(T, new_P, S)

    for p1 in 1:length(left_ids), s=1:S
        p0 = left_ids[p1]
        x_new[p1, s] = vp[s][p0]
    end

    x_new
end


"""
Transform a parameter vector to variational parameters in place.

Args:
 - xs: A (param x sources) matrix created from free variational parameters.
 - vp_free: Free variational parameters.  Only the ids in kept_ids
            will be updated.
 - kept_ids: Ids to update (from ids_free)

Returns:
 - Update vp_free in place.
"""
function array_to_free_vp!(xs::Array{T},
                           vp_free::FreeVariationalParams{T},
                           kept_ids::Vector{Int}) where {T}
    for s in 1:length(vp_free), p1 in 1:length(kept_ids)
        vp_free[s][kept_ids[p1]] = xs[p1, s]
    end
end


#########################
# Define the exported variables.

"""
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
bounds: The bounds for each parameter and each object in ElboArgs.
active_sources: The sources that are being optimized.    Only these sources'
    parameters are transformed into the parameter vector.
"""
struct DataTransform
    bounds::Vector{ParamBounds}
    active_sources::Vector{Int}
    S::Int
    # TODO: Maybe this should be initialized with ElboArgs with optional
    # custom bounds. Or maybe it should be part of ElboArgs with one transform
    # per celestial object rather than a single object containing an array of
    # transforms.
    function DataTransform(bounds::Vector{ParamBounds},
                           active_sources=collect(1:length(bounds)),
                           S=length(bounds))
        @assert length(bounds) == length(active_sources)
        @assert maximum(active_sources) <= S
        new(bounds, active_sources, S)
    end
end

function from_vp!(dt::DataTransform,
                  vp::VariationalParams{T},
                  vp_free::VariationalParams{T}) where {T}
    active_sources = dt.active_sources
    bounds = dt.bounds
    for i in eachindex(active_sources)
        vp_to_free!(vp[active_sources[i]], vp_free[i], bounds[i])
    end
    return nothing
end

function to_vp!(dt::DataTransform,
                vp_free::FreeVariationalParams{T},
                vp::VariationalParams{T}) where {T<:Number}
    active_sources = dt.active_sources
    bounds = dt.bounds
    for i in eachindex(active_sources)
        free_to_vp!(vp_free[i], vp[active_sources[i]], bounds[i])
    end
    return nothing
end

function from_vp(dt::DataTransform,
                 vp::VariationalParams{T}) where {T}
    vp_free = [zeros(T, length(ids_free)) for _ in 1:length(dt.active_sources)]
    from_vp!(dt, vp, vp_free)
    vp_free
end

function to_vp(dt::DataTransform, vp_free::FreeVariationalParams{T}) where {T}
    n_active_sources = length(dt.active_sources)
    @assert(n_active_sources == dt.S,
            string("to_vp is not supported when active_sources is a ",
                   "strict subset of all sources."))
    vp = [zeros(T, length(CanonicalParams)) for _ in 1:n_active_sources]
    to_vp!(dt, vp_free, vp)
    vp
end

function vp_to_array(dt::DataTransform,
                     vp::VariationalParams{T},
                     omitted_ids::Vector{Int}) where {T}
    vp_trans = from_vp(dt, vp)
    free_vp_to_array(vp_trans, omitted_ids)
end

function array_to_vp!(dt::DataTransform,
                      xs::Array{T},
                      vp::VariationalParams{T},
                      kept_ids::Vector{Int}) where {T}
    # This needs to update vp in place so that variables in omitted_ids
    # stay at their original values.
    vp_trans = from_vp(dt, vp)
    array_to_free_vp!(xs, vp_trans, kept_ids)
    to_vp!(dt, vp_trans, vp)
end


# Given a sensitive float with derivatives with respect to all the
# constrained parameters, calculate derivatives with respect to
# the unconstrained parameters.
#
# Note that all the other functions in DeterministicVI calculated derivatives with
# respect to the constrained parameterization.
function transform_sensitive_float!(dt::DataTransform,
                                    sf_free::SensitiveFloat,
                                    sf::SensitiveFloat,
                                    vp::VariationalParams{T},
                                    active_sources::Vector{Int}) where {T}

    if isnan(sf.v[])
       error("sf has NaN value:", sf_free.v[])
    end

    if any(isnan.(sf.d))
       error("sf has NaN derivatives:", sf_free.d[])
    end

    if any(isnan.(sf.h))
       error("sf has NaN hessian:", sf_free.h)
    end

    n_active_sources = length(active_sources)
    @assert size(sf.d) == (length(CanonicalParams), n_active_sources)

    transform_derivatives = get_transform_derivatives(vp, active_sources, dt.bounds)

    d, h = sf.d, sf.h
    free_d, free_h = sf_free.d, sf_free.h
    dparam_dfree = transform_derivatives.dparam_dfree
    d2param_dfree2 = transform_derivatives.d2param_dfree2

    sf_free.v[] = sf.v[]

    copy!(free_d, reshape(dparam_dfree' * d, length(UnconstrainedParams), n_active_sources))

    At_mul_B!(free_h, dparam_dfree, h * dparam_dfree)

    for i in eachindex(d)
        X, n = d2param_dfree2[i], d[i]
        for j in eachindex(X)
            free_h[j] += X[j] * n
        end
    end

    symmetrize!(free_h)

    if isnan(sf_free.v[])
        error("sf_free has NaN value:", sf_free.v[])
    end

    if any(isnan.(sf_free.d))
        error("sf_free has NaN derivatives:", sf_free.d[])
    end

    if any(isnan.(sf_free.h))
        error("sf_free has NaN hessian:", sf_free.h)
    end

    return sf_free
end

# Ensure exact symmetry, which is necessary for
# some numerical linear algebra routines.
function symmetrize!(A, c = 0.5)
    for i in 1:size(A, 1), j in 1:i
        A[i, j] = A[j, i] = c * (A[i, j] + A[j, i])
    end
    return A
end

function get_mp_transform(vp::VariationalParams{T},
                          active_sources::Vector{Int};
                          loc_scale=1.0,
                          loc_width=1e-4) where {T<:Number}
    bounds = Vector{ParamBounds}(length(active_sources))

    # Note that, for numerical reasons, the bounds must be on the scale
    # of reasonably meaningful changes.
    for si in 1:length(active_sources)
        s = active_sources[si]
        bounds[si] = ParamBounds()
        bounds[si][:pos] = Vector{ParamBox}(2)
        pos = vp[s][ids.pos]
        for axis in 1:2
            bounds[si][:pos][axis] =
                ParamBox(pos[axis] - loc_width, pos[axis] + loc_width, loc_scale)
        end
        bounds[si][:flux_loc] = Vector{ParamBox}(NUM_SOURCE_TYPES)
        bounds[si][:flux_scale] = Vector{ParamBox}(NUM_SOURCE_TYPES)
        for i in 1:NUM_SOURCE_TYPES
            bounds[si][:flux_loc][i] = ParamBox(-1.0, 10., 1.0)
            bounds[si][:flux_scale][i] = ParamBox(1e-4, 0.1, 1.0)
        end
        bounds[si][:color_mean] = Vector{ParamBox}(4 * NUM_SOURCE_TYPES)
        bounds[si][:color_var] = Vector{ParamBox}(4 * NUM_SOURCE_TYPES)
        for ind in 1:length(ids.color_mean)
            bounds[si][:color_mean][ind] = ParamBox(-10., 10., 1.0)
            bounds[si][:color_var][ind] = ParamBox(1e-4, 1., 1.0)
        end
        bounds[si][:gal_frac_dev] = ParamBox[ ParamBox(1e-2, 1 - 1e-2, 1.0) ]
        bounds[si][:gal_axis_ratio] = ParamBox[ ParamBox(1e-2, 1 - 1e-2, 1.0) ]
        bounds[si][:gal_angle] = ParamBox[ ParamBox(-10.0, 10.0, 1.0) ]
        bounds[si][:gal_radius_px] = ParamBox[ ParamBox(0.1, 70., 1.0) ]

        bounds[si][:a] = Vector{SimplexBox}(1)
        bounds[si][:a][1] = SimplexBox(0.005, 1.0, 2)

        bounds[si][:k] = Vector{SimplexBox}(NUM_SOURCE_TYPES)
        for i in 1:NUM_SOURCE_TYPES
            bounds[si][:k][i] = SimplexBox(0.01 / NUM_COLOR_COMPONENTS, 1.0, NUM_COLOR_COMPONENTS)
        end
    end

    DataTransform(bounds, active_sources, length(vp))
end


"""
Put the variational parameters within the bounds of the transform.
Returns:
  Updates vp in place.
"""
function enforce_bounds!(vp::VariationalParams{T},
                         active_sources::Vector{Int},
                         transform::DataTransform) where {T<:Number}
    for sa in eachindex(transform.active_sources), (param, constraint_vec) in transform.bounds[sa]
        s = active_sources[sa]
        is_box = isa(constraint_vec, Array{ParamBox})
        if is_box
            # Box parameters.
            for ind in 1:length(getfield(ids, param))
                constraint = constraint_vec[ind]
                if !(constraint.lb <=
                         vp[s][getfield(ids, param)[ind]] <=
                         constraint.ub)
                    # Don't set the value to exactly the lower bound to avoid Inf
                    diff = constraint.ub - constraint.lb
                    epsilon = diff == Inf ? 1e-12: diff * 1e-12
                    vp[s][getfield(ids, param)[ind]] =
                        min(vp[s][getfield(ids, param)[ind]], constraint.ub - epsilon)
                    vp[s][getfield(ids, param)[ind]] =
                        max(vp[s][getfield(ids, param)[ind]], constraint.lb + epsilon)
                end
            end
        else
            # Simplex parameters.
            param_size = size(getfield(ids, param))
            for col in 1:param_size[2]
                constraint = constraint_vec[col]
                param_sum = zero(T)
                for row in 1:param_size[1]
                    if !(constraint.lb <= vp[s][getfield(ids, param)[row, col]] <= 1.0)
                        # Don't set the value to exactly the lower bound to avoid Inf
                        epsilon = (1.0 - constraint.lb) * 1e-12
                        vp[s][getfield(ids, param)[row, col]] =
                            min(vp[s][getfield(ids, param)[row, col]], 1.0 - epsilon)
                        vp[s][getfield(ids, param)[row, col]] =
                            max(vp[s][getfield(ids, param)[row, col]],
                                    constraint.lb + epsilon)
                    end
                    param_sum += vp[s][getfield(ids, param)[row, col]]
                end
                if param_sum != 1.0
                    # Normalize in a way that maintains the lower bounds
                    rescale =
                      (1 - constraint.n * constraint.lb) /
                      (param_sum - constraint.n * constraint.lb)
                    vp[s][getfield(ids, param)[:, col]] =
                      constraint.lb +
                      rescale * (vp[s][getfield(ids, param)[:, col]] -
                                 constraint.lb)
                end
            end
        end
    end
end


end
