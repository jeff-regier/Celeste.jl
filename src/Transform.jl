# Convert between different parameterizations.

module Transform

using Celeste
using CelesteTypes

import Util

export pixel_rect_transform, world_rect_transform, free_transform, identity_transform, DataTransform

#export unconstrain_vp, rect_unconstrain_vp, constrain_vp, rect_constrain_vp
#export unconstrain_vp!, rect_unconstrain_vp!, constrain_vp!, rect_constrain_vp!
#export rect_unconstrain_sensitive_float, unconstrain_sensitive_float

type DataTransform
	# Functions to move between a ModelParameters object and a
	# transformation of the data for optimization.
    #
    # to_vp: A function that takes transformed parameters and returns variational parameters
    # from_vp: A function that takes variational parameters and returned transformed parameters
    # to_vp!: A function that takes (transformed paramters, variational parameters) and updates
    #   the variational parameters in place
    # from_vp!: A function that takes (variational paramters, transformed parameters) and updates
    #   the transformed parameters in place
    # ...
    # transform_sensitive_float: A function that takes (sensitive float, model parameters)
    #   where the sensitive float contains partial derivatives with respect to the
    #   variational parameters and returns a sensitive float with total derivatives with
    #   respect to the transformed parameters.

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

        function from_vp(vp::VariationalParams)
            S = length(vp)
            vp_free = [ zeros(id_size) for s = 1:S]
            from_vp!(vp, vp_free)
            vp_free
        end

        function to_vp(vp_free::FreeVariationalParams)
            S = length(vp_free)
            vp = [ zeros(length(CanonicalParams)) for s = 1:S]
            to_vp!(vp_free, vp)
            vp
        end

        function vp_to_vector(vp::VariationalParams, omitted_ids::Vector{Int64})
            vp_trans = from_vp(vp)
            trans_vp_to_vector(vp_trans, omitted_ids)
        end

        function vector_to_vp!(xs::Vector{Float64}, vp::VariationalParams,
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
function unchanged_vp!(vp::VariationalParams, new_vp::VariationalParams)
    # Leave the vp unchanged.
    S = length(vp)
    for s = 1:S
        new_vp[s][:] = vp[s]
    end
end

function unchanged_sensitive_float(sf::SensitiveFloat, mp::ModelParams)
    # Leave the sensitive float unchanged.
    deepcopy(sf)
end


#####################
# Conversion to and from vectors.

function unchanged_vp_to_vector(vp::VariationalParams, omitted_ids::Vector{Int64})
    # There is probably no use for this function, since you'll only be passing
    # trasformations to the optimizer, but I'll include it for completeness.
    error("Converting untransformed VarationalParams to a vector is not supported.")
end


function unchanged_vector_to_vp!(xs::Vector{Float64}, vp::VariationalParams,
                                 omitted_ids::Vector{Int64})
    # There is probably no use for this function, since you'll only be passing
    # trasformations to the optimizer, but I'll include it for completeness.
    error("Converting from a vector to untransformed VarationalParams is not supported.")
end


function free_vp_to_vector(vp::FreeVariationalParams, omitted_ids::Vector{Int64})
    # vp = variational parameters
    # omitted_ids = ids in ParamIndex
    #
    # There is probably no use for this function, since you'll only be passing
    # trasformations to the optimizer, but I'll include it for completeness.

    left_ids = setdiff(1:length(UnconstrainedParams), omitted_ids)
    new_P = length(left_ids)

    S = length(vp)
    vp_new = [zeros(new_P) for s in 1:S]

    for p1 in 1:length(left_ids)
        p0 = left_ids[p1]
        [ vp_new[s][p1] = vp[s][p0] for s in 1:S ]
    end

    reduce(vcat, vp_new)
end


function vector_to_free_vp!(xs::Vector{Float64}, vp_free::FreeVariationalParams,
                            omitted_ids::Vector{Int64})
    # xs: A vector created from free variational parameters.
    # free_vp: Free variational parameters.  Only the ids not in omitted_ids
    #   will be updated.
    # omitted_ids: Ids to omit (from ids_free)

    left_ids = setdiff(1:length(UnconstrainedParams), omitted_ids)

    P = length(left_ids)
    @assert length(xs) % P == 0
    S = int(length(xs) / P)
    xs2 = reshape(xs, P, S)

    for s in 1:S
        for p1 in 1:length(left_ids)
            p0 = left_ids[p1]
            vp_free[s][p0] = xs2[p1, s]
        end
    end
end



###############################################
# Functions for a "rectangular transform".  This matches the original Celeste
# script, contraining a to sum to one and scaling r1.

# Rescale some parameters to have similar dimensions to everything else.

# These are backwards.
const world_rect_rescaling = ones(length(UnconstrainedParams))
[world_rect_rescaling[id] *= 1e-3 for id in ids_free.r1]
[world_rect_rescaling[id] *= 1e5 for id in ids_free.u]

const pixel_rect_rescaling = ones(length(UnconstrainedParams))
[pixel_rect_rescaling[id] *= 1e-3 for id in ids_free.r1]

rect_unchanged_ids = [ "u", "r1", "r2",
                       "e_dev", "e_axis", "e_angle", "e_scale",
                       "c1", "c2"]

function vp_to_rect!(vp::VariationalParams, vp_free::RectVariationalParams,
                     rect_rescaling::Array{Float64, 1})
    # Convert a constrained to an unconstrained variational parameterization
    # that does not use logit or exp.

    S = length(vp)
    for s = 1:S
        # Variables that are unaffected by constraints (except for scaling):
        for id_string in rect_unchanged_ids
            id_symbol = convert(Symbol, id_string)
            vp_free[s][ids_free.(id_symbol)] = vp[s][ids.(id_symbol)]
        end

        # Simplicial constriants.  The original script used "a" to only
        # refer to the probability of being a galaxy, which is now the
        # second component of a.
        vp_free[s][ids_free.a[1]] = vp[s][ids.a[2]]

        # Keep all but the last row of k.
        vp_free[s][ids_free.k] = vp[s][ids.k[1:(Ia - 1), :]]

        for i in 1:length(ids_free)
            vp_free[s][i] = vp_free[s][i] .* rect_rescaling[i]
        end
    end
end

function rect_to_vp!(vp_free::RectVariationalParams, vp::VariationalParams,
                     rect_rescaling::Array{Float64, 1})
    # Convert an unconstrained to an constrained variational parameterization
    # where we don't use exp or logit.

    S = length(vp_free)
    for s = 1:S
        scaled_vp_free = deepcopy(vp_free[s])
        for i = 1:length(ids_free)
            scaled_vp_free[i] = scaled_vp_free[i] / rect_rescaling[i]
        end

        # Variables that are unaffected by constraints (except for scaling):
        for id_string in rect_unchanged_ids
            id_symbol = convert(Symbol, id_string)
            vp[s][ids.(id_symbol)] = scaled_vp_free[ids_free.(id_symbol)]
        end

        # Simplicial constriants.
        vp[s][ids.a[2]] = scaled_vp_free[ids_free.a[1]]
        vp[s][ids.a[1]] = 1.0 - vp[s][ids.a[2]]

        vp[s][ids.k[1, :]] = scaled_vp_free[ids_free.k]
        vp[s][ids.k[2, :]] = 1 - vp[s][ids.k[1, :]]
    end
end

function rect_unconstrain_sensitive_float(sf::SensitiveFloat, mp::ModelParams,
                                          rect_rescaling::Array{Float64, 1})
    # Given a sensitive float with derivatives with respect to all the
    # constrained parameters, calculate derivatives with respect to
    # the unconstrained parameters.
    #
    # Note that all the other functions in ElboDeriv calculated derivatives with
    # respect to the unconstrained parameterization.

    # Require that the input have all derivatives defined.
    @assert size(sf.d) == (length(CanonicalParams), mp.S)

    sf_free = zero_sensitive_float(UnconstrainedParams, mp.S)
    sf_free.v = sf.v

    for s in 1:mp.S
        # Variables that are unaffected by constraints (except for scaling):
        for id_string in rect_unchanged_ids
            id_symbol = convert(Symbol, id_string)

            # Flatten the indices for matrix indexing
            id_free_indices = reduce(vcat, ids_free.(id_symbol))
            id_indices = reduce(vcat, ids.(id_symbol))

            sf_free.d[id_free_indices, s] = sf.d[id_indices, s]
        end

        # Simplicial constriants.
        sf_free.d[ids_free.a[1], s] = sf.d[ids.a[2], s] - sf.d[ids.a[1], s]

        sf_free.d[collect(ids_free.k[1, :]), s] =
            sf.d[collect(ids.k[1, :]), s] - sf.d[collect(ids.k[2, :]), s]

        for i in 1:length(ids_free)
            sf_free.d[i, s] = sf_free.d[i, s] ./ rect_rescaling[i]
        end
    end

    sf_free
end


function pixel_vp_to_rect!(vp::VariationalParams, vp_free::RectVariationalParams)
    vp_to_rect!(vp, vp_free, pixel_rect_rescaling)
end

function world_vp_to_rect!(vp::VariationalParams, vp_free::RectVariationalParams)
    vp_to_rect!(vp, vp_free, world_rect_rescaling)
end

function pixel_rect_to_vp!(vp_free::RectVariationalParams, vp::VariationalParams)
    rect_to_vp!(vp_free, vp, pixel_rect_rescaling)
end

function world_rect_to_vp!(vp_free::RectVariationalParams, vp::VariationalParams)
    rect_to_vp!(vp_free, vp, world_rect_rescaling)
end

function pixel_rect_unconstrain_sensitive_float(sf::SensitiveFloat, mp::ModelParams)
    rect_unconstrain_sensitive_float(sf, mp, pixel_rect_rescaling)
end

function world_rect_unconstrain_sensitive_float(sf::SensitiveFloat, mp::ModelParams)
    rect_unconstrain_sensitive_float(sf, mp, world_rect_rescaling)
end


###############################################
# Functions for a "free transform".  Eventually the idea is that this will
# have every parameter completely unconstrained.
free_unchanged_ids = [ "u", "e_dev", "e_axis", "e_angle", "e_scale",
                       "c1", "c2"]

function vp_to_free!(vp::VariationalParams, vp_free::FreeVariationalParams)
    # Convert a constrained to an unconstrained variational parameterization
    # on all of the real line.
    S = length(vp)
    for s = 1:S
        # Variables that are unaffected by constraints:
        for id_string in free_unchanged_ids
            id_symbol = convert(Symbol, id_string)
            vp_free[s][ids_free.(id_symbol)] = vp[s][ids.(id_symbol)]
        end

        # Simplicial constriants.  The original script used "a" to only
        # refer to the probability of being a galaxy, which is now the
        # second component of a.
        vp_free[s][ids_free.a[1]] = Util.inv_logit(vp[s][ids.a[2]])

        vp_free[s][ids_free.e_dev] = Util.inv_logit(vp[s][ids_free.e_dev])

        # In contrast, the original script used the last component of k
        # as the free parameter.
        vp_free[s][ids_free.k[1, :]] = Util.inv_logit(vp[s][ids.k[1, :]])

	# e_axis is not technically a simplicial constraint but it must lie in (0, 1).
        vp_free[s][ids_free.e_axis] = Util.inv_logit(vp[s][ids.e_axis])

        # Positivity constraints
        vp_free[s][ids_free.e_scale] = log(vp[s][ids.e_scale])
        vp_free[s][ids_free.c2] = log(vp[s][ids.c2])

	# Parameterize brightness in more observable quantities.
        # ids_free.r1 = log(r1) + log(r2)
        # ids_free.r2 = log(r1) + 2 log(r2)
        vp_free[s][ids_free.r1] = log(vp[s][ids.r1]) + log(vp[s][ids.r2])
        vp_free[s][ids_free.r2] = log(vp[s][ids.r1]) + 2 * log(vp[s][ids.r2]) 
    end
end

function free_to_vp!(vp_free::FreeVariationalParams, vp::VariationalParams)
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

    	vp[s][ids.e_dev] = Util.logit(vp_free[s][ids_free.e_dev])

        vp[s][ids.k[1, :]] = Util.logit(vp_free[s][ids_free.k[1, :]])
        vp[s][ids.k[2, :]] = 1.0 - vp[s][ids.k[1, :]]
	
	# e_axis is not technically a simplicial constraint but it must lie in (0, 1).
        vp[s][ids.e_axis] = Util.logit(vp_free[s][ids_free.e_axis])

         # Positivity constraints
        vp[s][ids.e_scale] = exp(vp_free[s][ids_free.e_scale])
        vp[s][ids.c2] = exp(vp_free[s][ids_free.c2])

        # Brightness
        vp[s][ids.r1] = exp(2 * vp_free[s][ids_free.r1] - vp_free[s][ids_free.r2])
        vp[s][ids.r2] = exp(vp_free[s][ids_free.r2] - vp_free[s][ids_free.r1])
    end
end


function free_unconstrain_sensitive_float(sf::SensitiveFloat, mp::ModelParams)
    # Given a sensitive float with derivatives with respect to all the
    # constrained parameters, calculate derivatives with respect to
    # the unconstrained parameters.
    #
    # Note that all the other functions in ElboDeriv calculated derivatives with
    # respect to the unconstrained parameterization.

    # Require that the input have all derivatives defined.
    @assert size(sf.d) == (length(CanonicalParams), mp.S)

    sf_free = zero_sensitive_float(UnconstrainedParams, mp.S)
    sf_free.v = sf.v

    for s in 1:mp.S
        # Variables that are unaffected by constraints:
        for id_string in rect_unchanged_ids
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

        # Positivity constraints.
        sf_free.d[ids_free.e_scale, s] = sf.d[ids.e_scale, s] .* mp.vp[s][ids.e_scale]
        sf_free.d[collect(ids_free.c2), s] =
            sf.d[collect(ids.c2), s] .* mp.vp[s][collect(ids.c2)]

        # Brightness.
        sf_free.d[ids_free.r1, s] =
            2.0 * sf.d[ids.r1, s] .* mp.vp[s][ids.r1] - sf.d[ids.r2, s] .* mp.vp[s][ids.r2]
        sf_free.d[ids_free.r2, s] =
            -sf.d[ids.r1, s] .* mp.vp[s][ids.r1] + sf.d[ids.r2, s] .* mp.vp[s][ids.r2]
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
