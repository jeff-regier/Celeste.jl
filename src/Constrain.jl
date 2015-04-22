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

	DataTransform(to_vp!, from_vp!, transform_sensitive_float) = begin

        from_vp = function(vp::VariationalParams)
            S = length(vp)
            vp_free = [ zeros(ids_free.size) for s = 1:S]
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

function unchanged_vp!(vp::VariationalParams, new_vp::VariationalParams)
    # Leave the vp unchanged.
    S = length(vp_free)
    for s = 1:S
        new_vp[s][all_params] = vp[s][all_params]
    end
end

function unchanged_sensitive_float(sf::SensitiveFloat, mp::ModelParams)
    # Leave the sensitive float unchanged.
    deepcopy(sf)
end


function rect_constrain_vp!(vp_free::RectVariationalParams, vp::VariationalParams)
    # Convert an unconstrained to an constrained variational parameterization
    # where we don't use exp or logit.

    S = length(vp_free)
    for s = 1:S
        # The default is everything being the same.

        # Maybe something like this instead:
        #
        # for id_symbol in names(ids)
        #     if id_symbol != convert(Symbol, chi)
        #         vp[s][ids.(id_symbol)] = vp_free[s][ids_free.(id_symbol)]
        # end

        vp[s][ids.mu] = vp_free[s][ids_free.mu]
        vp[s][ids.theta] = vp_free[s][ids_free.theta]
        vp[s][ids.rho] = vp_free[s][ids_free.rho]
        vp[s][ids.phi] = vp_free[s][ids_free.phi]
        vp[s][ids.sigma] = vp_free[s][ids_free.sigma]
        vp[s][ids.kappa] = vp_free[s][ids_free.kappa]
        vp[s][ids.beta] = vp_free[s][ids_free.beta]
        vp[s][ids.lambda] = vp_free[s][ids_free.lambda]
        vp[s][ids.gamma] = vp_free[s][ids_free.gamma]
        vp[s][ids.zeta] = vp_free[s][ids_free.zeta]

        # Simplicial constriants.
        vp[s][ids.chi[2]] = vp_free[s][ids_free.chi[1]]
        vp[s][ids.chi[1]] = 1.0 - vp[s][ids.chi[2]]
    end
end


function rect_unconstrain_vp!(vp::VariationalParams, vp_free::RectVariationalParams)
    # Convert a constrained to an unconstrained variational parameterization
    # that does not use logit or exp.

    S = length(vp)
    for s = 1:S
        # Variables that are unaffected by constraints:
        vp_free[s][ids_free.mu] = vp[s][ids.mu]
        vp_free[s][ids_free.theta] = vp[s][ids.theta]
        vp_free[s][ids_free.rho] = vp[s][ids.rho]
        vp_free[s][ids_free.phi] = vp[s][ids.phi]
        vp_free[s][ids_free.sigma] = vp[s][ids.sigma]
        vp_free[s][ids_free.kappa] = vp[s][ids.kappa]
        vp_free[s][ids_free.beta] = vp[s][ids.beta]
        vp_free[s][ids_free.lambda] = vp[s][ids.lambda]
        vp_free[s][ids_free.gamma] = vp[s][ids.gamma]
        vp_free[s][ids_free.zeta] = vp[s][ids.zeta]

        # Simplicial constriants.  The original script used "chi" to only
        # refer to the probability of being a galaxy, which is now the
        # second component of chi.
        vp_free[s][ids_free.chi[1]] = vp[s][ids.chi[2]]
    end
end

function rect_constrain_vp!(vp_free::RectVariationalParams, vp::VariationalParams)
    # Convert an unconstrained to an constrained variational parameterization
    # where we don't use exp or logit.

    S = length(vp_free)
    for s = 1:S
        # The default is everything being the same.

        # Maybe something like this instead:
        #
        # for id_symbol in names(ids)
        #     if id_symbol != convert(Symbol, chi)
        #         vp[s][ids.(id_symbol)] = vp_free[s][ids_free.(id_symbol)]
        # end

        vp[s][ids.mu] = vp_free[s][ids_free.mu]
        vp[s][ids.theta] = vp_free[s][ids_free.theta]
        vp[s][ids.rho] = vp_free[s][ids_free.rho]
        vp[s][ids.phi] = vp_free[s][ids_free.phi]
        vp[s][ids.sigma] = vp_free[s][ids_free.sigma]
        vp[s][ids.kappa] = vp_free[s][ids_free.kappa]
        vp[s][ids.beta] = vp_free[s][ids_free.beta]
        vp[s][ids.lambda] = vp_free[s][ids_free.lambda]
        vp[s][ids.gamma] = vp_free[s][ids_free.gamma]
        vp[s][ids.zeta] = vp_free[s][ids_free.zeta]

        # Simplicial constriants.
        vp[s][ids.chi[2]] = vp_free[s][ids_free.chi[1]]
        vp[s][ids.chi[1]] = 1.0 - vp[s][ids.chi[2]]
    end
end


function free_unconstrain_vp!(vp::VariationalParams, vp_free::FreeVariationalParams)
    # Convert a constrained to an unconstrained variational parameterization
    # on all of the real line.
    S = length(vp)
    for s = 1:S
        # Variables that are unaffected by constraints:
        vp_free[s][ids_free.mu] = vp[s][ids.mu]
        vp_free[s][ids_free.theta] = vp[s][ids.theta]
        vp_free[s][ids_free.rho] = vp[s][ids.rho]
        vp_free[s][ids_free.phi] = vp[s][ids.phi]
        vp_free[s][ids_free.sigma] = vp[s][ids.sigma]
        vp_free[s][ids_free.kappa] = vp[s][ids.kappa]
        vp_free[s][ids_free.beta] = vp[s][ids.beta]
        vp_free[s][ids_free.lambda] = vp[s][ids.lambda]

        # Simplicial constriants.  The original script used "chi" to only
        # refer to the probability of being a galaxy, which is now the
        # second component of chi.
        vp_free[s][ids_free.chi[1]] = Util.inv_logit(vp[s][ids.chi[2]])

        # Positivity constraints
        vp_free[s][ids_free.gamma] = log(vp[s][ids.gamma])
        vp_free[s][ids_free.zeta] = log(vp[s][ids.zeta]) 
    end
end

function free_constrain_vp!(vp_free::FreeVariationalParams, vp::VariationalParams)
    # Convert an unconstrained to an constrained variational parameterization.
    S = length(vp_free)
    for s = 1:S
        # The default is everything being the same.
        vp[s][ids.mu] = vp_free[s][ids_free.mu]
        vp[s][ids.theta] = vp_free[s][ids_free.theta]
        vp[s][ids.rho] = vp_free[s][ids_free.rho]
        vp[s][ids.phi] = vp_free[s][ids_free.phi]
        vp[s][ids.sigma] = vp_free[s][ids_free.sigma]
        vp[s][ids.kappa] = vp_free[s][ids_free.kappa]
        vp[s][ids.beta] = vp_free[s][ids_free.beta]
        vp[s][ids.lambda] = vp_free[s][ids_free.lambda]

        # Simplicial constriants.
        vp[s][ids.chi[2]] = Util.logit(vp_free[s][ids_free.chi[1]])
        vp[s][ids.chi[1]] = 1.0 - vp[s][ids.chi[2]]

         # Positivity constraints
        vp[s][ids.gamma] = exp(vp_free[s][ids_free.gamma])
        vp[s][ids.zeta] = exp(vp_free[s][ids_free.zeta])
    end
end

# Conversion functions for sensitive floats.
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
        # Unless specifically transformed, the derivatives are unchanged.
        sf_free.d[ids_free.mu, s] = sf.d[ids.mu, s]
        sf_free.d[ids_free.theta, s] = sf.d[ids.theta, s]
        sf_free.d[ids_free.rho, s] = sf.d[ids.rho, s]
        sf_free.d[ids_free.phi, s] = sf.d[ids.phi, s]
        sf_free.d[ids_free.sigma, s] = sf.d[ids.sigma, s]
        sf_free.d[reduce(vcat, ids_free.kappa), s] =
            sf.d[reduce(vcat, ids.kappa), s]
        sf_free.d[reduce(vcat, ids_free.beta), s] =
            sf.d[reduce(vcat, ids.beta), s]
        sf_free.d[reduce(vcat, ids_free.lambda), s] =
            sf.d[reduce(vcat, ids.lambda), s]
        sf_free.d[ids_free.gamma, s] = sf.d[ids.gamma, s]
        sf_free.d[ids_free.zeta, s] = sf.d[ids.zeta, s]

        # Simplicial constriants.
        sf_free.d[ids_free.chi[1], s] = sf.d[ids.chi[2], s] - sf.d[ids.chi[1], s]
    end

    sf_free
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
        # Unless specifically transformed, the derivatives are unchanged.
        sf_free.d[ids_free.mu, s] = sf.d[ids.mu, s]
        sf_free.d[ids_free.theta, s] = sf.d[ids.theta, s]
        sf_free.d[ids_free.rho, s] = sf.d[ids.rho, s]
        sf_free.d[ids_free.phi, s] = sf.d[ids.phi, s]
        sf_free.d[ids_free.sigma, s] = sf.d[ids.sigma, s]
        sf_free.d[reduce(vcat, ids_free.kappa), s] =
            sf.d[reduce(vcat, ids.kappa), s]
        sf_free.d[reduce(vcat, ids_free.beta), s] =
            sf.d[reduce(vcat, ids.beta), s]
        sf_free.d[reduce(vcat, ids_free.lambda), s] =
            sf.d[reduce(vcat, ids.lambda), s]

        # TODO: write in general form.  Note that the old "chi" is now chi[2].
        # Simplicial constriants.
        this_chi = mp.vp[s][ids.chi[2]]
        sf_free.d[ids_free.chi[1], s] =
            (sf.d[ids.chi[2], s] - sf.d[ids.chi[1], s]) * this_chi * (1.0 - this_chi)

        # Positivity constraints.
        sf_free.d[ids_free.gamma, s] = sf.d[ids.gamma, s] .* mp.vp[s][ids.gamma]
        sf_free.d[ids_free.zeta, s] = sf.d[ids.zeta, s] .* mp.vp[s][ids.zeta]
    end

    sf_free
end

rect_transform = DataTransform(rect_constrain_vp!, rect_unconstrain_vp!,
                               rect_unconstrain_sensitive_float)

free_transform = DataTransform(free_constrain_vp!, free_unconstrain_vp!,
                               free_unconstrain_sensitive_float)

identity_transform = DataTransform(unchanged_vp!, unchanged_vp!,
                                   unchanged_sensitive_float)

end