# Calculate the Kullback-Leibler divergences between pairs of well-known
# parameteric distributions, and derivatives with respect to the parameters
# of the first distributions.

# Note that KL divergences may be between parameters of two different types,
# e.g. if the prior is a float and the parameter is a dual number.

import ..SensitiveFloats: SensitiveFloat

######################################################
# KL divergences

"""
KL divergence between a pair of beta distributions

Args:
- alpha2: the shape parameter for the second beta distribution
- beta2: the scale parameter for the second beta distribution

Returns:
- a function the takes shape and scale parameters for the first
  beta distribution, and returns the KL divergence and its derivatives
"""
function gen_beta_kl{NumType <: Number}(alpha2::NumType, beta2::NumType)
    const lgamma_alpha2 = lgamma(alpha2)
    const lgamma_beta2 = lgamma(beta2)
    function this_beta_kl{NumType2 <: Number}(
            alpha1::NumType2, beta1::NumType2, calculate_derivs::Bool)

        alpha_diff = alpha1 - alpha2
        beta_diff = beta1 - beta2
        both_inv_diff = -(alpha_diff + beta_diff)
        di_both1 = digamma(alpha1 + beta1)

        log_term = lgamma(alpha1 + beta1) - lgamma(alpha1) - lgamma(beta1)
        log_term -= lgamma(alpha2 + beta2) - lgamma_alpha2 - lgamma_beta2
        apart_term = alpha_diff * digamma(alpha1) + beta_diff * digamma(beta1)
        together_term = both_inv_diff * di_both1
        kl = log_term + apart_term + together_term

        grad = zeros(NumType2, 2)
        hess = zeros(NumType2, 2, 2)

        if calculate_derivs
            trigamma_alpha1 = trigamma(alpha1)
            trigamma_beta1 = trigamma(beta1)
            trigamma_both = trigamma(alpha1 + beta1)
            grad[1] = alpha_diff * trigamma_alpha1 + both_inv_diff * trigamma_both
            grad[2] = beta_diff * trigamma_beta1 + both_inv_diff * trigamma_both

            quadgamma_both = polygamma(2, alpha1 + beta1)
            hess[1, 1] = alpha_diff * polygamma(2, alpha1) +
                         both_inv_diff * quadgamma_both +
                         trigamma_alpha1 - trigamma_both
            hess[2, 2] = beta_diff * polygamma(2, beta1) +
                         both_inv_diff * quadgamma_both +
                         trigamma_beta1 - trigamma_both
            hess[1, 2] = hess[2, 1] =
                -trigamma_both + both_inv_diff * quadgamma_both
        end

        return kl, grad, hess
    end
end


"""
KL divergence between a pair of categorical distributions.
"""
function gen_categorical_kl{NumType <: Number}(p2::Vector{NumType})
    function this_categorical_kl{NumType2 <: Number}(
            p1::Vector{NumType2}, calculate_derivs::Bool)

        kl = zero(NumType2)
        grad = zeros(NumType2, length(p1))
        hess = zeros(NumType2, length(p1), length(p1))

        for i in 1:length(p1)
            log_ratio = log(p1[i]) - log(p2[i])
            kl += p1[i] * log_ratio
            if calculate_derivs
                grad[i] = 1 + log_ratio
                hess[i, i] = 1 / p1[i]
            end
        end

        return kl, grad, hess
    end
end


"""
KL divergence between a pair of univariate Gaussian distributions

Args:
- mu2: the mean for the second Gaussian distribution
- beta2: the variance for the second Gaussian distribution

Returns:
- a function the takes mean and variance parameters for the first
  Gaussian distribution, and returns the KL divergence and
  its derivatives.  The indexing of the derivatives is mean first then variance.
"""
function gen_normal_kl{NumType <: Number}(mu2::NumType, sigma2Sq::NumType)
    const log_sigma2Sq = log(sigma2Sq)
    const precision2 = 1 / sigma2Sq
    function this_normal_kl{NumType2 <: Number}(
            mu1::NumType2, sigma1Sq::NumType2, calculate_derivs::Bool)
        diff = mu1 - mu2
        kl = .5 * (log_sigma2Sq - log(sigma1Sq) +
                   (sigma1Sq + (diff)^2) / sigma2Sq - 1)

        grad = zeros(NumType2, 2)
        hess = zeros(NumType2, 2, 2)
        if calculate_derivs
            grad[1] = precision2 * diff                 # Gradient wrt the mean
            grad[2] = 0.5 * (precision2 - 1 / sigma1Sq) # Gradient wrt the var
            hess[1, 1] = precision2
            hess[2, 2] = 0.5 / (sigma1Sq ^ 2)
        end
        return kl, grad, hess
    end
end


"""
KL divergence between a pair of multivariate normal distributions,
the first having a diagonal covariance matrix

Args:
- mean2: the mean for the second normal distribution
- cov2: the covariance matrix for the second normal distribution

Returns:
- a function the takes mean and variance parameters for the first normal
  distribution, and returns the KL divergence and its derivatives
"""
function gen_diagmvn_mvn_kl{NumType <: Number}(
  mean2::Vector{NumType}, cov2::Matrix{NumType})
    const precision2 = cov2^-1
    const logdet_cov2 = logdet(cov2)
    const K = length(mean2)

    function this_diagmvn_mvn_kl{NumType2 <: Number}(
        mean1::Vector{NumType2}, vars1::Vector{NumType2}, calculate_derivs::Bool)

      diff = mean2 - mean1

      kl = sum(diag(precision2) .* vars1) - K
      kl += (diff' * precision2 * diff)[]
      kl += -sum(log(vars1)) + logdet_cov2
      kl = 0.5 * kl

      grad_mean = zeros(NumType2, K)
      grad_var = zeros(NumType2, K)
      hess_mean = zeros(NumType2, K, K)
      hess_var = zeros(NumType2, K, K)

      if calculate_derivs
          grad_mean = -1 * precision2 * diff
          grad_var = 0.5 * (diag(precision2) - 1 ./ vars1)

          hess_mean = precision2
          for k in 1:K
              hess_var[k, k] = 0.5 ./ (vars1[k] ^ 2)
          end
      end

      return kl, grad_mean, grad_var, hess_mean, hess_var
    end
end


#######################################################
# Functions to subtract single-source KL divergences from a SensitiveFloat

"""
A sensitive float representing a single term a[i] which can be combined
with other sensitive floats.
"""
function get_a_term_sensitive_float{NumType <: Number}(
        a::NumType, i::Integer, calculate_derivs::Bool)
    a_term = zero_sensitive_float(CanonicalParams, NumType)
    a_term.v[1] = a
    if calculate_derivs
        a_term.d[ids.a[i], 1] = 1
    end
    return a_term
end


"""
A sensitive float representing a single term k[d, i] which can be combined
with other sensitive floats.
"""
function get_k_term_sensitive_float{NumType <: Number}(
        k::NumType, i::Integer, d::Integer, calculate_derivs::Bool)

    k_term = zero_sensitive_float(CanonicalParams, NumType)
    k_term.v[1] = k
    if calculate_derivs
        k_term.d[ids.k[d, i], 1] = 1
    end
    return k_term
end


"""
Subtract the KL divergence from the prior for c
"""
function subtract_kl_c!{NumType <: Number}(
    vs::Vector{NumType}, kl_source::SensitiveFloat{CanonicalParams, NumType},
    calculate_derivs::Bool)

    kl_term = zero_sensitive_float(CanonicalParams, NumType)
    for i in 1:Ia, d in 1:D
        clear!(kl_term)
        pp_kl_cid = gen_diagmvn_mvn_kl(
            prior.c_mean[:, d, i], prior.c_cov[:, :, d, i])
        mean_ids = ids.c1[:, i]
        var_ids = ids.c2[:, i]
        kl, grad_mean, grad_var, hess_mean, hess_var =
            pp_kl_cid(vs[mean_ids], vs[var_ids], calculate_derivs)
        kl_term.v[1] = kl

        if calculate_derivs
            kl_term.d[mean_ids, 1] = grad_mean
            kl_term.d[var_ids, 1] = grad_var
            kl_term.h[mean_ids, mean_ids] = hess_mean
            kl_term.h[var_ids, var_ids] = hess_var
        end

        a_term = get_a_term_sensitive_float(vs[ids.a[i]], i, calculate_derivs)
        k_term = get_k_term_sensitive_float(vs[ids.k[d, i]], i, d, calculate_derivs)
        multiply_sfs!(kl_term, a_term, calculate_derivs)
        multiply_sfs!(kl_term, k_term, calculate_derivs)
        add_scaled_sfs!(kl_source, kl_term, -1.0, calculate_derivs)
    end
end


"""
Subtract the KL divergence from the prior for k.

Args:
    vs: Variational parameters for a source
    kl_k: Updated in place.  A SensitiveFloat containing the KL divergence
          of the k parameters.
"""
function subtract_kl_k!{NumType <: Number}(
        vs::Vector{NumType},
        kl_source::SensitiveFloat{CanonicalParams, NumType},
        calculate_derivs::Bool)

    kl_term = zero_sensitive_float(CanonicalParams, NumType)
    for i in 1:Ia
        clear!(kl_term)
        k_ind = ids.k[:, i]
        pp_kl_ki = gen_categorical_kl(prior.k[:, i])
        kl, grad, hess = pp_kl_ki(vs[k_ind], calculate_derivs)
        kl_term.v[1] = kl
        if calculate_derivs
            kl_term.d[k_ind, 1] = grad
            kl_term.h[k_ind, k_ind] = hess
        end
        a_term = get_a_term_sensitive_float(vs[ids.a[i]], i, calculate_derivs)
        multiply_sfs!(kl_term, a_term, calculate_derivs)
        add_scaled_sfs!(kl_source, kl_term, -1.0, calculate_derivs)
    end
end


"""
Subtract the KL divergence from the prior for r for object type i.
"""
function subtract_kl_r!{NumType <: Number}(
        vs::Vector{NumType}, kl_source::SensitiveFloat{CanonicalParams, NumType},
        calculate_derivs::Bool)

    kl_term = zero_sensitive_float(CanonicalParams, NumType)
    for i in 1:Ia
        clear!(kl_term)
        pp_kl_r = gen_normal_kl(prior.r_mean[i], prior.r_var[i])
        kl, grad, hess = pp_kl_r(vs[ids.r1[i]], vs[ids.r2[i]], calculate_derivs)
        r_ind = Integer[ ids.r1[i], ids.r2[i] ]
        kl_term.v[1] = kl
        if calculate_derivs
            kl_term.d[r_ind, 1] = grad
            kl_term.h[r_ind, r_ind] = hess
        end
        a_term = get_a_term_sensitive_float(vs[ids.a[i]], i, calculate_derivs)
        multiply_sfs!(kl_term, a_term, calculate_derivs)
        add_scaled_sfs!(kl_source, kl_term, -1.0, calculate_derivs)
    end
end


"""
Subtract the KL divergence from the prior for a
"""
function subtract_kl_a!{NumType <: Number}(
        vs::Vector{NumType}, kl_source::SensitiveFloat{CanonicalParams, NumType},
        calculate_derivs::Bool)

    pp_kl_a = gen_categorical_kl(prior.a)

    kl, grad, hess = pp_kl_a(vs[ids.a], calculate_derivs)
    kl_source.v[1] -= kl
    if calculate_derivs
        kl_source.d[ids.a, 1] -= grad
        kl_source.h[ids.a, ids.a] -= hess
    end
end


"""
Subtract the KL divergences for all sources.
"""
function subtract_kl!{NumType <: Number}(
        ea::ElboArgs{NumType}, accum::SensitiveFloat{CanonicalParams, NumType};
        calculate_derivs::Bool=true)

    for sa in 1:length(ea.active_sources)
        s = ea.active_sources[sa]
        kl_source = zero_sensitive_float(CanonicalParams, NumType)
        subtract_kl_a!(ea.vp[s], kl_source, calculate_derivs)
        subtract_kl_k!(ea.vp[s], kl_source, calculate_derivs)
        subtract_kl_r!(ea.vp[s], kl_source, calculate_derivs)
        subtract_kl_c!(ea.vp[s], kl_source, calculate_derivs)

        add_sources_sf!(accum, kl_source, sa, calculate_derivs)
    end

end
