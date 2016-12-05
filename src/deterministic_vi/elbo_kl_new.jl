# Calculate the Kullback-Leibler divergences between pairs of well-known parameteric
# distributions, and derivatives with respect to the parameters of the first distributions.

######################
# Beta Distributions #
######################

"""
Returns the KL divergence between a pair of beta distributions

Args:
- α₁: the shape parameter for the first beta distribution
- β₁: the scale parameter for the first beta distribution
- α₂: the shape parameter for the second beta distribution
- β₂: the scale parameter for the second beta distribution
"""
function beta_kl(α₁, β₁, α₂, β₂)
    α₁_plus_β₁ = α₁ + β₁
    α₂_plus_β₂ = α₂ + β₂
    α₁_minus_α₂ = α₁ - α₂
    β₁_minus_β₂ = β₁ - β₂
    log_term = lgamma(α₁_plus_β₁) - lgamma(α₁) - lgamma(β₁)
    log_term -= lgamma(α₂_plus_β₂) - lgamma(α₂) - lgamma(β₂)
    apart_term = α₁_minus_α₂ * digamma(α₁) + β₁_minus_β₂ * digamma(β₁)
    together_term = -(α₁_minus_α₂ + β₁_minus_β₂) * digamma(α₁_plus_β₁)
    return log_term + apart_term + together_term
end

#############################
# Categorical Distributions #
#############################

"""
Returns the KL divergence between a pair of categorical distributions
"""
categorical_kl(p₁, p₂) = sum(a * (log(a) - log(b)) for (a, b) in zip(p₁, p₂))

##########################
# Gaussian Distributions #
##########################

"""
Returns the KL divergence between a pair of univariate Gaussian distributions

Args:
- μ₁: the mean for the first Gaussian distribution
- var₁: the variance for the first Gaussian distribution
- μ₂: the mean for the second Gaussian distribution
- var₂: the variance for the second Gaussian distribution
"""
gaussian_kl(μ₁, var₁, μ₂, var₂) = .5 * (log(var₂) - log(var₁) + (var₁ + (μ₁ - μ₂)^2) / var₂ - 1)

#####################################
# Multivariate Normal Distributions #
#####################################

"""
Returns the KL divergence between a pair of multivariate normal distributions,
the first having a diagonal covariance matrix

Args:
- μ₁: the mean for the first normal distribution
- var₁: the variance parameters for the first normal distribution
- μ₂: the mean for the second normal distribution
- Σ₂: the covariance matrix for the second normal distribution
"""
function diagmvn_mvn_kl(μ₁, var₁, μ₂, Σ₂, inv_Σ₂ = inv(Σ₂), logdet_Σ₂ = logdet(Σ₂))
    μ₂_minus_μ₁ = μ₂ - μ₁
    kl = sum(diag(inv_Σ₂) .* var₁) - length(μ₂)
    kl += dot(μ₂_minus_μ₁, inv_Σ₂ * μ₂_minus_μ₁)
    kl += logdet_Σ₂ - sum(log.(var₁))
    return 0.5 * kl
end

###########################################
# Subtracting KL divergences from sources #
###########################################

subtract_kl_a(vs) = -(categorical_kl(vs[ids.a[:, 1]], prior.a))

function subtract_kl_r(vs)
    kl = zero(eltype(vs))
    for i in 1:Ia
        μ₁, var₁ = vs[ids.r1[i]], vs[ids.r2[i]]
        μ₂, var₂ = prior.r_mean[i], prior.r_var[i]
        kl -= vs[ids.a[i, 1]] * gaussian_kl(μ₁, var₁, μ₂, var₂)
    end
    return kl
end

function subtract_kl_k(vs)
    kl = zero(eltype(vs))
    for i in 1:Ia
        kl -= vs[ids.a[i, 1]] * categorical_kl(vs[ids.k[:, i]], prior.k[:, i])
    end
    return kl
end

function subtract_kl_c(vs)
    kl = zero(eltype(vs))
    for i in 1:Ia
        μ₁, var₁ = vs[ids.c1[:, i]], vs[ids.c2[:, i]]
        a = vs[ids.a[i, 1]]
        for d in 1:D
            μ₂, Σ₂ = prior.c_mean[:, d, i], prior.c_cov[:, :, d, i]
            kl -= a * vs[ids.k[d, i]] * diagmvn_mvn_kl(μ₁, var₁, μ₂, Σ₂)
        end
    end
    return kl
end

"""
Subtract the KL divergences for a single source.
"""
function subtract_kl(vs)
    kl = zero(eltype(vs))
    kl += subtract_kl_a(vs)
    kl += subtract_kl_k(vs)
    kl += subtract_kl_r(vs)
    kl += subtract_kl_c(vs)
    return kl
end

###################
# Differentiation #
###################

const PARAM_LENGTH = length(CanonicalParams)
const CHUNK_SIZE = ForwardDiff.pickchunksize(PARAM_LENGTH)
const DUAL_TYPE = ForwardDiff.Dual{CHUNK_SIZE,Float64}
const NESTED_KL_GRADIENT_BUFFER = zeros(DUAL_TYPE,PARAM_LENGTH)
const NESTED_KL_JACOBIAN_CONFIG = ForwardDiff.JacobianConfig{CHUNK_SIZE}(rand(PARAM_LENGTH))

const kl_gradient! = ReverseDiff.compile_gradient(subtract_kl, rand(PARAM_LENGTH))
const nested_kl_gradient! = ReverseDiff.compile_gradient(subtract_kl, rand(DUAL_TYPE,PARAM_LENGTH))

nested_kl_gradient(x) = nested_kl_gradient!(NESTED_KL_GRADIENT_BUFFER, x)
kl_hessian!(out, x) = ForwardDiff.jacobian!(out, nested_kl_gradient, x, NESTED_KL_JACOBIAN_CONFIG)

###############
# Entry Point #
###############

function subtract_kl_source!(kl_source::SensitiveFloat, vs, kl_grad, kl_hess)
    if kl_source.has_gradient
        kl_gradient!(kl_grad, vs)
        kl_source.v[] = DiffBase.value(kl_grad)
        copy!(kl_source.d, DiffBase.gradient(kl_grad))
    else
        kl_source.v[] = subtract_kl(vs)
    end
    if kl_source.has_hessian
        kl_hessian!(kl_hess, vs)
        copy!(kl_source.h, kl_hess)
    end
    return kl_source
end

function subtract_kl_all_sources!(ea::ElboArgs,
                                  accum::SensitiveFloat,
                                  kl_source::SensitiveFloat,
                                  kl_grad, kl_hess)
    for sa in 1:length(ea.active_sources)
        subtract_kl_source!(kl_source, ea.vp[ea.active_sources[sa]], kl_grad, kl_hess)
        add_sources_sf!(accum, kl_source, sa)
    end
    return accum
end
