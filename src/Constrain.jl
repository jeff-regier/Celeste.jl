# Convert between different parameterizations.

module Constrain

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
    # transform_sensitive_float: A function that takes (sensitive float, model parameters)
    #   where the sensitive float contains partial derivatives with respect to the
    #   variational parameters and returns a sensitive float with total derivatives with
    #   respect to the transformed parameters.

	to_vp::Function
	from_vp::Function
	to_vp!::Function
	from_vp!::Function
	transform_sensitive_float::Function

	DataTransform(to_vp!, from_vp!, transform_sensitive_float, id_size::Integer) = begin

        from_vp = function(vp::VariationalParams)
            S = length(vp)
            vp_free = [ zeros(id_size) for s = 1:S]
            from_vp!(vp, vp_free)
            vp_free
        end

        to_vp = function(vp_free::FreeVariationalParams)
            S = length(vp_free)
            vp = [ zeros(ids.size) for s = 1:S]
            to_vp!(vp_free, vp)
            vp
        end

		new(to_vp, from_vp, to_vp!, from_vp!,
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

        # Keep all but the last column of kappa.
        vp_free[s][ids_free.kappa] = vp[s][ids.kappa[:, 1:(Ia - 1)]]
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

        vp[s][ids.kappa[:, 1]] = vp_free[s][ids_free.kappa]
        vp[s][ids.kappa[:, 2]] = 1 - vp[s][ids.kappa[:, 1]]
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
        sf_free.d[ids_free.kappa[:, 1], s] = sf.d[ids.kappa[:, 1], s] - sf.d[ids.kappa[:, 2], s]
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
        vp_free[s][ids_free.kappa[:, 1]] = Util.inv_logit(vp[s][ids.kappa[:, 1]])

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

        vp[s][ids.kappa[:, 1]] = Util.logit(vp_free[s][ids_free.kappa[:, 1]])
        vp[s][ids.kappa[:, 2]] = 1.0 - vp[s][ids.kappa[:, 1]]

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

        this_kappa = mp.vp[s][ids.kappa[:, 1]]
        sf_free.d[ids_free.kappa[:, 1], s] =
            (sf.d[ids.kappa[:, 1], s] - sf.d[ids.kappa[:, 2], s]) .*
            this_kappa .* (1.0 - this_kappa)

        # Positivity constraints.
        sf_free.d[ids_free.gamma, s] = sf.d[ids.gamma, s] .* mp.vp[s][ids.gamma]
        sf_free.d[ids_free.zeta, s] = sf.d[ids.zeta, s] .* mp.vp[s][ids.zeta]
    end

    sf_free
end

rect_transform = DataTransform(rect_to_vp!, vp_to_rect!,
                               rect_unconstrain_sensitive_float, ids_free.size)

free_transform = DataTransform(free_to_vp!, vp_to_free!,
                               free_unconstrain_sensitive_float, ids_free.size)

identity_transform = DataTransform(unchanged_vp!, unchanged_vp!,
                                   unchanged_sensitive_float, ids.size)

end