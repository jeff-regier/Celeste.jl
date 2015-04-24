# Convert between different parameterizations.

module Transform

using Celeste
using CelesteTypes

import Util

export rect_transform, free_transform, identity_transform, DataTransform

#export unconstrain_vp, rect_unconstrain_vp, constrain_vp, rect_constrain_vp
#export unconstrain_vp!, rect_unconstrain_vp!, constrain_vp!, rect_constrain_vp!
#export rect_unconstrain_sensitive_float, unconstrain_sensitive_float

type DataTransform
	# Functiones to move between a ModelParameters object and a
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
            vp = [ zeros(ids.size) for s = 1:S]
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
        new_vp[s][all_params] = vp[s][all_params]
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

    left_ids = setdiff(all_params_free, omitted_ids)
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

    left_ids = setdiff(all_params_free, omitted_ids)

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
# script, contraining chi to sum to one and scaling gamma.

const rect_rescaling = ones(length(all_params))

# Rescale some parameters to have similar dimensions to everything else.
[rect_rescaling[id] *= 1e-3 for id in ids.gamma]

rect_unchanged_ids = [ "mu", "gamma", "zeta",
                       "theta", "rho", "phi", "sigma",
                       "beta", "lambda"]

function vp_to_rect!(vp::VariationalParams, vp_free::RectVariationalParams)
    # Convert a constrained to an unconstrained variational parameterization
    # that does not use logit or exp.

    S = length(vp)
    for s = 1:S
        # Variables that are unaffected by constraints (except for scaling):
        for id_string in rect_unchanged_ids
            id_symbol = convert(Symbol, id_string)
            vp_free[s][ids_free.(id_symbol)] =
                (vp[s][ids.(id_symbol)] .* rect_rescaling[ids.(id_symbol)])
        end

        # Simplicial constriants.  The original script used "chi" to only
        # refer to the probability of being a galaxy, which is now the
        # second component of chi.
        vp_free[s][ids_free.chi[1]] = vp[s][ids.chi[2]]

        # Keep all but the last row of kappa.
        vp_free[s][ids_free.kappa] = vp[s][ids.kappa[1:(Ia - 1), :]]
    end
end

function rect_to_vp!(vp_free::RectVariationalParams, vp::VariationalParams)
    # Convert an unconstrained to an constrained variational parameterization
    # where we don't use exp or logit.

    S = length(vp_free)
    for s = 1:S
        # For unchanged ids, simply scale them.
        for id_string in rect_unchanged_ids
            id_symbol = convert(Symbol, id_string)
            vp[s][ids.(id_symbol)] =
                (vp_free[s][ids_free.(id_symbol)] ./ rect_rescaling[ids.(id_symbol)])
        end

        # Simplicial constriants.
        vp[s][ids.chi[2]] = vp_free[s][ids_free.chi[1]]
        vp[s][ids.chi[1]] = 1.0 - vp[s][ids.chi[2]]

        vp[s][ids.kappa[1, :]] = vp_free[s][ids_free.kappa]
        vp[s][ids.kappa[2, :]] = 1 - vp[s][ids.kappa[1, :]]
    end
end

function rect_unconstrain_sensitive_float(sf::SensitiveFloat, mp::ModelParams)
    # Given a sensitive float with derivatives with respect to all the
    # constrained parameters, calculate derivatives with respect to
    # the unconstrained parameters.
    #
    # Note that all the other functions in ElboDeriv calculated derivatives with
    # respect to the unconstrained parameterization.

    # Require that the input have all derivatives defined.
    @assert size(sf.d) == (ids.size, mp.S)

    sf_free = zero_sensitive_float(collect(1:mp.S), CelesteTypes.all_params_free)
    sf_free.v = sf.v

    # Currently the param_index is only really used within ElboDeriv.  By the
    # time the data hits the optimizer, we assume everything has a derivative. 
     sf_free.param_index = all_params_free

    for s in 1:mp.S
        # Variables that are unaffected by constraints (except for scaling):
        for id_string in rect_unchanged_ids
            id_symbol = convert(Symbol, id_string)

            # Flatten the indices for matrix indexing
            id_free_indices = reduce(vcat, ids_free.(id_symbol))
            id_indices = reduce(vcat, ids.(id_symbol))

            sf_free.d[id_free_indices, s] =
                (sf.d[id_indices, s] ./ rect_rescaling[id_indices])
        end

        # Simplicial constriants.
        sf_free.d[ids_free.chi[1], s] = sf.d[ids.chi[2], s] - sf.d[ids.chi[1], s]

        sf_free.d[collect(ids_free.kappa[1, :]), s] =
            sf.d[collect(ids.kappa[1, :]), s] - sf.d[collect(ids.kappa[2, :]), s]
    end

    sf_free
end

###############################################
# Functions for a "free transform".  Eventually the idea is that this will
# have every parameter completely unconstrained.
free_unchanged_ids = [ "mu", "theta", "rho", "phi", "sigma",
                       "beta", "lambda"]

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

        # Simplicial constriants.  The original script used "chi" to only
        # refer to the probability of being a galaxy, which is now the
        # second component of chi.
        vp_free[s][ids_free.chi[1]] = Util.inv_logit(vp[s][ids.chi[2]])

        # In contrast, the original script used the last component of kappa
        # as the free parameter.
        vp_free[s][ids_free.kappa[1, :]] = Util.inv_logit(vp[s][ids.kappa[1, :]])

        # Positivity constraints
        vp_free[s][ids_free.gamma] = log(vp[s][ids.gamma])
        vp_free[s][ids_free.zeta] = log(vp[s][ids.zeta]) 
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
        vp[s][ids.chi[2]] = Util.logit(vp_free[s][ids_free.chi[1]])
        vp[s][ids.chi[1]] = 1.0 - vp[s][ids.chi[2]]

        vp[s][ids.kappa[1, :]] = Util.logit(vp_free[s][ids_free.kappa[1, :]])
        vp[s][ids.kappa[2, :]] = 1.0 - vp[s][ids.kappa[1, :]]

         # Positivity constraints
        vp[s][ids.gamma] = exp(vp_free[s][ids_free.gamma])
        vp[s][ids.zeta] = exp(vp_free[s][ids_free.zeta])
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
    @assert size(sf.d) == (ids.size, mp.S)

    sf_free = zero_sensitive_float(collect(1:mp.S), CelesteTypes.all_params_free)
    sf_free.v = sf.v

    # Currently the param_index is only really used within ElboDeriv.  By the
    # time the data hits the optimizer, we assume everything has a derivative. 
    sf_free.param_index = all_params_free

    for s in 1:mp.S
        # Variables that are unaffected by constraints:
        for id_string in rect_unchanged_ids
            id_symbol = convert(Symbol, id_string)

            # Flatten the indices for matrix indexing
            id_free_indices = reduce(vcat, ids_free.(id_symbol))
            id_indices = reduce(vcat, ids.(id_symbol))

            sf_free.d[id_free_indices, s] = sf.d[id_indices, s]
        end

        # TODO: write in general form.  Note that the old "chi" is now chi[2].
        # Simplicial constriants.
        this_chi = mp.vp[s][ids.chi[2]]
        sf_free.d[ids_free.chi[1], s] =
            (sf.d[ids.chi[2], s] - sf.d[ids.chi[1], s]) * this_chi * (1.0 - this_chi)

        this_kappa = collect(mp.vp[s][ids.kappa[1, :]])
        sf_free.d[collect(ids_free.kappa[1, :]), s] =
            (sf.d[collect(ids.kappa[1, :]), s] - sf.d[collect(ids.kappa[2, :]), s]) .*
            this_kappa .* (1.0 - this_kappa)

        # Positivity constraints.
        sf_free.d[ids_free.gamma, s] = sf.d[ids.gamma, s] .* mp.vp[s][ids.gamma]
        sf_free.d[ids_free.zeta, s] = sf.d[ids.zeta, s] .* mp.vp[s][ids.zeta]
    end

    sf_free
end

rect_transform = DataTransform(rect_to_vp!, vp_to_rect!,
                               vector_to_free_vp!, free_vp_to_vector, 
                               rect_unconstrain_sensitive_float,
                               ids_free.size)

free_transform = DataTransform(free_to_vp!, vp_to_free!,
                               vector_to_free_vp!, free_vp_to_vector,
                               free_unconstrain_sensitive_float, ids_free.size)

identity_transform = DataTransform(unchanged_vp!, unchanged_vp!,
                                   unchanged_vector_to_vp!, unchanged_vp_to_vector,
                                   unchanged_sensitive_float, ids.size)

end