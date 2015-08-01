# Convert between different parameterizations.

module Transform

using Celeste
using CelesteTypes

import Util
VERSION < v"0.4.0-dev" && using Docile

export pixel_rect_transform, world_rect_transform, free_transform, identity_transform, DataTransform


#export unconstrain_vp, rect_unconstrain_vp, constrain_vp, rect_constrain_vp
#export unconstrain_vp!, rect_unconstrain_vp!, constrain_vp!, rect_constrain_vp!
#export rect_unconstrain_sensitive_float, unconstrain_sensitive_float

@doc """
Functions to move between a single source's variational parameters and a
transformation of the data for optimization.

to_vp: A function that takes transformed parameters and returns variational parameters
from_vp: A function that takes variational parameters and returned transformed parameters
to_vp!: A function that takes (transformed paramters, variational parameters) and updates
  the variational parameters in place
from_vp!: A function that takes (variational paramters, transformed parameters) and updates
  the transformed parameters in place
...
transform_sensitive_float: A function that takes (sensitive float, model parameters)
  where the sensitive float contains partial derivatives with respect to the
  variational parameters and returns a sensitive float with total derivatives with
  respect to the transformed parameters. """ ->
type DataTransform

	to_vp::Function
	from_vp::Function
	to_vp!::Function
	from_vp!::Function
    vp_to_vector::Function
    vector_to_vp!::Function
	transform_sensitive_float::Function

	DataTransform(to_vp!::Function, from_vp!::Function,
                  vector_to_trans_vp!::Function, trans_vp_to_vector::Function,
                  transform_sensitive_float::Function, id_size::Integer) = begin

        function from_vp{NumType <: Number}(vp::Vector{NumType})
            vp_free = zeros(NumType, id_size)
            from_vp!(vp, vp_free)
            vp_free
        end

        function to_vp{NumType <: Number}(vp_free::Vector{NumType})
            vp = zeros(length(CanonicalParams))
            to_vp!(vp_free, vp)
            vp
        end

        function vp_to_vector{NumType <: Number}(vp::Vector{NumType},
                                                 omitted_ids::Vector{Int64})
            vp_trans = from_vp(vp)
            trans_vp_to_vector(vp_trans, omitted_ids)
        end

        function vector_to_vp!{NumType <: Number}(xs::Vector{NumType},
                                                  vp::Vector{NumType},
                                                  omitted_ids::Vector{Int64})
            # This needs to update vp in place so that variables in omitted_ids
            # stay at their original values.
            vp_trans = from_vp(vp)
            vector_to_trans_vp!(xs, vp_trans, omitted_ids)
            to_vp!(vp_trans, vp)
        end

		new(to_vp, from_vp, to_vp!, from_vp!, vp_to_vector, vector_to_vp!,
            transform_sensitive_float)
	end
end

###############################################
# Functions for an identity transform.
function unchanged_vp!{NumType <: Number}(vp::Vector{NumType}, new_vp::Vector{NumType})
    new_vp[:] = vp
end

function unchanged_sensitive_float{NumType <: Number}(sf::SensitiveFloat, vp::Vector{NumType})
    # Leave the sensitive float unchanged.
    deepcopy(sf)
end


#####################
# Conversion to and from vectors.

function unchanged_vp_to_vector{NumType <: Number}(vp::Vector{NumType}, omitted_ids::Vector{Int64})
    # There is probably no use for this function, since you'll only be passing
    # trasformations to the optimizer, but I'll include it for completeness.
    error("Converting untransformed VarationalParams to a vector is not supported.")
end


function unchanged_vector_to_vp!{NumType <: Number}(xs::Vector{Float64}, vp::Vector{NumType},
                                                    omitted_ids::Vector{Int64})
    # There is probably no use for this function, since you'll only be passing
    # trasformations to the optimizer, but I'll include it for completeness.
    error("Converting from a vector to untransformed VarationalParams is not supported.")
end


function free_vp_to_vector{NumType <: Number}(vp::Vector{NumType},
                                              omitted_ids::Vector{Int64})
    # vp = variational parameters
    # omitted_ids = ids in ParamIndex
    #
    # There is probably no use for this function, since you'll only be passing
    # trasformations to the optimizer, but I'll include it for completeness.

    left_ids = setdiff(1:length(UnconstrainedParams), omitted_ids)
    new_p = 1:length(left_ids)
    new_p[:] = vp[left_ids]
end


function vector_to_free_vp!{NumType <: Number}(xs::Vector{NumType},
                                               vp_free::Vector{NumType},
                                               omitted_ids::Vector{Int64})
    # xs: A vector created from free variational parameters.
    # free_vp: Free variational parameters.  Only the ids not in omitted_ids
    #   will be updated.
    # omitted_ids: Ids to omit (from ids_free)

    left_ids = setdiff(1:length(UnconstrainedParams), omitted_ids)
    vp_free[left_ids] = xs
end


###############################################
# Functions for a "free transform".

function unbox_parameter(param::Union(Float64, Vector{Float64}), upper_bound::Float64, lower_bound::Float64)
    @assert(all(lower_bound .< param .< upper_bound), "param outside bounds: $param ($lower_bound, $upper_bound)")
    param_scaled = (param - lower_bound) / (upper_bound - lower_bound)
    Util.inv_logit(param_scaled)
end

function box_parameter(free_param::Union(Float64, Vector{Float64}), upper_bound::Float64, lower_bound::Float64)
    Util.logit(free_params) * (upper_bound - lower_bound) + lower_bound
end

function unbox_sensitive_float!(sf::SensitiveFloat,
                                param::Union(Float64, Vector{Float64}), upper_bound::Float64, lower_bound::Float64,
                                index::Union(Int64, Vector{Int64}), s::Int64)
    @assert length(param) == length(index)

    # Box constraints.  Strict inequality is not required for derivatives.
    @assert(all(lower_bound .<= param .<= upper_bound), "param outside bounds: $param ($lower_bound, $upper_bound)")
    param_scaled = (param - lower_bound) ./ (upper_bound - lower_bound)
    sf.d[index, s] = sf.d[index, s] .* param_scaled .* (1 - param_scaled) .* (upper_bound - lower_bound)
end


# Just for example.  In real life each source gets its own location constraints.
typealias ParamBounds Dict{Symbol, (Float64, Float64)}
bounds = ParamBounds()
bounds[:u] = (-1., 1.)
bounds[:r1] = (1e-4, 1e12)
bounds[:r2] = (1e-4, 0.1)
bounds[:c1] = (-10., 10.)
bounds[:c2] = (1e-4, 1.)
bounds[:e_dev] = (1e-2, 1 - 1e-2)
bounds[:e_axis] = (1e-4, 1 - 1e-4)
bounds[:e_angle] = (-1e10, 1e10)
bounds[:e_scale] = (0.2, 15)



@doc """
Convert a variational parameter vector to an unconstrained version using
the lower bounds lbs and ubs (which are expressed)
""" ->
function vp_to_free!{NumType <: Number}(vp::Vector{NumType},
                                        vp_free::Vector{NumType},
                                        bounds::ParamBounds)
    # Simplicial constriants.  The original script used "a" to only
    # refer to the probability of being a galaxy, which is now the
    # second component of a.
    vp_free[ids_free.a[1]] = Util.inv_logit(vp[ids.a[2]])

    # In contrast, the original script used the last component of k
    # as the free parameter.
    vp_free[ids_free.k[1, :]] = Util.inv_logit(vp[ids.k[1, :]])

    # Box constraints.
    for param, limits in bounds
        vp_free[ids_free.(param)] = unbox_parameter(vp[ids.(param)], limits[1], limits[2])
    end
end


function free_to_vp!{NumType <: Number}(vp_free::FreeVariationalParams{NumType},
                                        vp::VariationalParams{NumType})
    # Convert an unconstrained to an constrained variational parameterization.
    S = length(vp_free)
    for s = 1:S
                # Variables that are unaffected by constraints:
        for id_string in free_unchanged_ids
            id_symbol = convert(Symbol, id_string)
            vp[s][ids.(id_symbol)] = vp_free[s][ids_free.(id_symbol)]
        end

        # Simplicial constriants.
        vp[s][ids.a[2]] = Util.logit(vp_free[s][ids_free.a[1]])
        vp[s][ids.a[1]] = 1.0 - vp[s][ids.a[2]]

        vp[s][ids.k[1, :]] = Util.logit(vp_free[s][ids_free.k[1, :]])
        vp[s][ids.k[2, :]] = 1.0 - vp[s][ids.k[1, :]]
	
    	# [0, 1] constraints.
        vp[s][ids.e_dev] = Util.logit(vp_free[s][ids_free.e_dev])
        vp[s][ids.e_axis] = Util.logit(vp_free[s][ids_free.e_axis])

        # Positivity constraints
        vp[s][ids.e_scale] = exp(vp_free[s][ids_free.e_scale])
        vp[s][ids.c2] = exp(vp_free[s][ids_free.c2])

        # Box constraints.
        vp[s][ids.r1] = Util.logit(vp_free[s][ids.r1]) * (free_r1_max - free_r1_min) + free_r1_min
        vp[s][ids.r2] = Util.logit(vp_free[s][ids.r2]) * (free_r2_max - free_r2_min) + free_r2_min
    end
end


function free_unconstrain_sensitive_float{NumType <: Number}(sf::SensitiveFloat, mp::ModelParams{NumType})
    # Given a sensitive float with derivatives with respect to all the
    # constrained parameters, calculate derivatives with respect to
    # the unconstrained parameters.
    #
    # Note that all the other functions in ElboDeriv calculated derivatives with
    # respect to the unconstrained parameterization.

    # Require that the input have all derivatives defined.
    @assert size(sf.d) == (length(CanonicalParams), mp.S)

    sf_free = zero_sensitive_float(UnconstrainedParams, NumType, mp.S)
    sf_free.v = sf.v

    for s in 1:mp.S
        # Variables that are unaffected by constraints:
        for id_string in free_unchanged_ids
            id_symbol = convert(Symbol, id_string)

            # Flatten the indices for matrix indexing
            id_free_indices = reduce(vcat, ids_free.(id_symbol))
            id_indices = reduce(vcat, ids.(id_symbol))

            sf_free.d[id_free_indices, s] = sf.d[id_indices, s]
        end

        # TODO: write in general form.  Note that the old "a" is now a[2].
        # Simplicial constriants.
        this_a = mp.vp[s][ids.a[2]]
        sf_free.d[ids_free.a[1], s] =
            (sf.d[ids.a[2], s] - sf.d[ids.a[1], s]) * this_a * (1.0 - this_a)

        this_k = collect(mp.vp[s][ids.k[1, :]])
        sf_free.d[collect(ids_free.k[1, :]), s] =
            (sf.d[collect(ids.k[1, :]), s] - sf.d[collect(ids.k[2, :]), s]) .*
            this_k .* (1.0 - this_k)

        # [0, 1] constraints.
        this_dev = mp.vp[s][ids.e_dev]
        sf_free.d[ids_free.e_dev, s] = sf.d[ids.e_dev, s] * this_dev * (1.0 - this_dev)

        this_axis = mp.vp[s][ids.e_axis]
        sf_free.d[ids_free.e_axis, s] = sf.d[ids.e_axis, s] * this_axis * (1.0 - this_axis)

        # Positivity constraints.
        sf_free.d[ids_free.e_scale, s] = sf.d[ids.e_scale, s] .* mp.vp[s][ids.e_scale]
        sf_free.d[collect(ids_free.c2), s] = sf.d[collect(ids.c2), s] .* mp.vp[s][collect(ids.c2)]

        # Box constraints.  Strict inequality is not required for derivatives.
        @assert(all(free_r1_min .<= mp.vp[s][ids.r1] .<= free_r1_max),
                "r1 outside bounds for $s: $(mp.vp[s][ids.r1])")
        @assert(all(free_r2_min .<= mp.vp[s][ids.r2] .<= free_r2_max),
                "r2 outside bounds for $s: $(mp.vp[s][ids.r2])")
        free_r1_scaled = (mp.vp[s][ids.r1] - free_r1_min) / (free_r1_max - free_r1_min)
        free_r2_scaled = (mp.vp[s][ids.r2] - free_r2_min) / (free_r2_max - free_r2_min)
        sf_free.d[ids_free.r1, s] = sf.d[ids.r1, s] .* free_r1_scaled .* (1 - free_r1_scaled) .* (free_r1_max - free_r1_min)
        sf_free.d[ids_free.r2, s] = sf.d[ids.r2, s] .* free_r2_scaled .* (1 - free_r2_scaled) .* (free_r2_max - free_r2_min)
    end

    sf_free
end

#########################
# Define the exported variables.

pixel_rect_transform = DataTransform(pixel_rect_to_vp!, pixel_vp_to_rect!,
                                     vector_to_free_vp!, free_vp_to_vector,
                                     pixel_rect_unconstrain_sensitive_float,
                                     length(UnconstrainedParams))

world_rect_transform = DataTransform(world_rect_to_vp!, world_vp_to_rect!,
                                     vector_to_free_vp!, free_vp_to_vector,
                                     world_rect_unconstrain_sensitive_float,
                                     length(UnconstrainedParams))

free_transform = DataTransform(free_to_vp!, vp_to_free!,
                               vector_to_free_vp!, free_vp_to_vector,
                               free_unconstrain_sensitive_float,
                               length(UnconstrainedParams))

identity_transform = DataTransform(unchanged_vp!, unchanged_vp!,
                                   unchanged_vector_to_vp!, unchanged_vp_to_vector,
                                   unchanged_sensitive_float,
                                   length(CanonicalParams))

end
