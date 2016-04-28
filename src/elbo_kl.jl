# Calculate the Kullback-Leibler divergences between pairs of well-known
# parameteric distributions, and derivatives with respect to the parameters
# of the first distributions.

# Note that KL divergences may be between parameters of two different types,
# e.g. if the prior is a float and the parameter is a dual number.


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
    function this_beta_kl{NumType2 <: Number}(alpha1::NumType2, beta1::NumType2)
        alpha_diff = alpha1 - alpha2
        beta_diff = beta1 - beta2
        both_inv_diff = -(alpha_diff + beta_diff)
        di_both1 = digamma(alpha1 + beta1)

        log_term = lgamma(alpha1 + beta1) - lgamma(alpha1) - lgamma(beta1)
        log_term -= lgamma(alpha2 + beta2) - lgamma(alpha2) - lgamma(beta2)
        apart_term = alpha_diff * digamma(alpha1) + beta_diff * digamma(beta1)
        together_term = both_inv_diff * di_both1
        log_term + apart_term + together_term
    end
end


function gen_categorical_kl{NumType <: Number}(p2::Vector{NumType})
    function this_categorical_kl{NumType2 <: Number}(p1::Vector{NumType2})
        v = zero(NumType2)

        for i in 1:length(p2)
            log_ratio = log(p1[i]) - log(p2[i])
            v += p1[i] * log_ratio
        end

        v
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
  its derivatives
"""
function gen_normal_kl{NumType <: Number}(mu2::NumType, sigma2Sq::NumType)
    const log_sigma2Sq = log(sigma2Sq)
    function this_normal_lk{NumType2 <: Number}(mu1::NumType2, sigma1Sq::NumType2)
        diff = mu1 - mu2
        .5 * ((log_sigma2Sq - log(sigma1Sq)) +
            (sigma1Sq + (diff)^2) / sigma2Sq - 1)
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
      mean1::Vector{NumType2}, vars1::Vector{NumType2})
        # const diag_precision2 = convert(Vector{NumType2}, diag(precision2))
        diff = mean2 - mean1

        v = sum(diag(precision2) .* vars1) - K
        v += (diff' * precision2 * diff)[]
        v += -sum(log(vars1)) + logdet_cov2
        0.5v
    end
end


"""
Subtract the KL divergence from the prior for c
"""
function subtract_kl_c{NumType <: Number}(
    d::Int, i::Int, vs::Vector{NumType}, pp::PriorParams)

  a = vs[ids.a[i]]
  k = vs[ids.k[d, i]]

  pp_kl_cid = gen_diagmvn_mvn_kl(pp.c_mean[:, d, i], pp.c_cov[:, :, d, i])
  -pp_kl_cid(vs[ids.c1[:, i]], vs[ids.c2[:, i]]) * a * k
end


"""
Subtract the KL divergence from the prior for k
"""
function subtract_kl_k{NumType <: Number}(
  i::Int, vs::Vector{NumType}, pp::PriorParams)

    pp_kl_ki = gen_categorical_kl(pp.k[:, i])
    -vs[ids.a[i]] * pp_kl_ki(vs[ids.k[:, i]])
end


"""
Subtract the KL divergence from the prior for r for object type i.
"""
function subtract_kl_r{NumType <: Number}(
  i::Int, vs::Vector{NumType}, pp::PriorParams)
    a = vs[ids.a[i]]
    pp_kl_r = gen_normal_kl(pp.r_mean[i], pp.r_var[i])
    v = pp_kl_r(vs[ids.r1[i]], vs[ids.r2[i]])
    -a * v
end


"""
Subtract the KL divergence from the prior for a
"""
function subtract_kl_a{NumType <: Number}(vs::Vector{NumType}, pp::PriorParams)
    pp_kl_a = gen_categorical_kl(pp.a)
    -pp_kl_a(vs[ids.a])
end


function subtract_kl!{NumType <: Number}(
        mp::ModelParams{NumType}, accum::SensitiveFloat{CanonicalParams, NumType};
        calculate_derivs::Bool=true, calculate_hessian::Bool=true)

    # The KL divergence as a function of the active source variational parameters.
    function subtract_kl_value_wrapper{NumType2 <: Number}(vp_vec::Vector{NumType2})
        elbo_val = zero(NumType2)
        vp_active = reshape(vp_vec, length(CanonicalParams), length(mp.active_sources))
        for sa in 1:length(mp.active_sources)
            vs = vp_active[:, sa]
            elbo_val += subtract_kl_a(vs, mp.pp)

            for i in 1:Ia
                    elbo_val += subtract_kl_r(i, vs, mp.pp)
                    elbo_val += subtract_kl_k(i, vs, mp.pp)
                    for d in 1:D
                        elbo_val += subtract_kl_c(d, i, vs, mp.pp)
                    end
            end
        end
        elbo_val
    end

    vp_vec = reduce(vcat, Vector{NumType}[ mp.vp[sa] for sa in mp.active_sources ])

    const P = length(CanonicalParams)
    Sa = length(mp.active_sources)

    if calculate_derivs
        if calculate_hessian
            hess, all_results =
                ForwardDiff.hessian(subtract_kl_value_wrapper,
                                    vp_vec,
                                    ForwardDiff.AllResults)
            accum.h += hess
            accum.d += reshape(ForwardDiff.gradient(all_results), P, Sa);
            accum.v[1] += ForwardDiff.value(all_results)
        else
            grad, all_results =
                ForwardDiff.gradient(subtract_kl_value_wrapper,
                                     vp_vec,
                                     ForwardDiff.AllResults)
            accum.d += reshape(grad, P, Sa);
            accum.v[1] += ForwardDiff.value(all_results)
        end
    else
        accum.v[1] += subtract_kl_value_wrapper(vp_vec)
    end
end

