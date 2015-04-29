module KL


export gen_beta_kl, gen_isobvnormal_improper
export gen_wrappedcauchy_uniform_kl, gen_categorical_kl, gen_gamma_kl
export gen_normal_kl, gen_isobvnormal_kl, gen_diagmvn_mvn_kl


trigamma(x) = polygamma(1, x)


function gen_beta_kl(alpha2::Float64, beta2::Float64)
    function(alpha1, beta1)
        alpha_diff = alpha1 - alpha2
        beta_diff = beta1 - beta2
        both_inv_diff = -(alpha_diff + beta_diff)
        di_both1 = digamma(alpha1 + beta1)
        tri_both1 = trigamma(alpha1 + beta1)

        log_term = lgamma(alpha1 + beta1) - lgamma(alpha1) - lgamma(beta1)
        log_term -= lgamma(alpha2 + beta2) - lgamma(alpha2) - lgamma(beta2)
        apart_term = alpha_diff * digamma(alpha1) + beta_diff * digamma(beta1)
        together_term = both_inv_diff * di_both1
        v = log_term + apart_term + together_term

        d_alpha1 = alpha_diff * trigamma(alpha1)
        d_alpha1 += both_inv_diff * tri_both1

        d_beta1 = beta_diff * trigamma(beta1)
        d_beta1 += both_inv_diff * tri_both1

        v, (d_alpha1, d_beta1)
    end
end


function gen_wrappedcauchy_uniform_kl()
    function(scale1::Float64)
        v = -log(1 - exp(-2scale1))
        d_scale1 = -2exp(-2scale1) / (1 - exp(-2scale1))
        v, (d_scale1,)
    end
end


function gen_categorical_kl(p2::Vector{Float64})
    function(p1::Vector{Float64})
        v = 0.
        d_p1 = Array(Float64, length(p1))

        for i in 1:length(p2)
            log_ratio = log(p1[i]) - log(p2[i])
            v += p1[i] * log_ratio
            d_p1[i] = 1 + log_ratio
        end

        v, (d_p1,)
    end
end


function gen_gamma_kl(k2::Float64, theta2::Float64)
    function(k1::Float64, theta1::Float64)
        digamma_k1 = digamma(k1)
        theta_ratio = (theta1 - theta2) / theta2
        shape_diff = k1 - k2

        v = shape_diff * digamma_k1
        v += -lgamma(k1) + lgamma(k2)
        v += k2 * (log(theta2) - log(theta1))
        v += k1 * theta_ratio

        d_k1 = shape_diff * trigamma(k1)
        d_k1 += theta_ratio

        d_theta1 = -k2 / theta1
        d_theta1 += k1 / theta2

        v, (d_k1, d_theta1)
    end
end


function gen_normal_kl(mu2::Float64, sigma2Sq::Float64)
    log_sigma2Sq = log(sigma2Sq)
    function(mu1::Float64, sigma1Sq::Float64)
        diff = mu1 - mu2
        v = .5 * ((log_sigma2Sq - log(sigma1Sq)) + (sigma1Sq + (diff)^2) / sigma2Sq - 1)
        d_mu1 = diff / sigma2Sq
        d_sigma1Sq = 0.5 * (-1. / sigma1Sq + 1 / sigma2Sq)
        v, (d_mu1, d_sigma1Sq)
    end
end


function gen_isobvnormal_kl(mean2::Vector{Float64}, var2::Float64)
    function(mean1::Vector{Float64}, var1::Float64)
        diff_sq = (mean1[1] - mean2[1])^2 + (mean1[2] - mean2[2])^2
        v = var1 / var2 + diff_sq / 2var2 - 1 + log(var2 / var1)

        d_mean1 = (mean1 .- mean2) ./ var2
        d_var1 = 1 / var2 - 1 / var1

        v, (d_mean1, d_var1)
    end
end


function gen_diagmvn_mvn_kl(mean2::Vector{Float64}, cov2::Matrix{Float64})
    const precision2 = cov2^-1
    const logdet_cov2 = logdet(cov2)
    const K = length(mean2)

    function(mean1::Vector{Float64}, vars1::Vector{Float64})
        diff = mean2 - mean1

        v = sum(diag(precision2) .* vars1) - K
        v += (diff' * precision2 * diff)[]
        v += -sum(log(vars1)) + logdet_cov2
        v *= 0.5

        d_mean1 = precision2 * -diff
        d_vars1 = 0.5 * diag(precision2)
        d_vars1[:] += -0.5 ./ vars1

        v, (d_mean1, d_vars1)
    end
end


function gen_isobvnormal_flat_kl()
    function(var1::Float64)
        v = -(1 + log(2pi) + log(var1))
        d_var1 = -1 / var1
        v, (d_var1,)
    end
end

end

