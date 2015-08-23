using Celeste
using Base.Test
using Distributions
import GSL.deriv_central

println("Running KL tests.")

function verify_kl(q_dist, p_dist, claimed_kl::Float64)
    sample_size = 4_000_000
    q_samples = rand(q_dist, sample_size)
    empirical_kl_samples = logpdf(q_dist, q_samples) - logpdf(p_dist, q_samples)
    empirical_kl = mean(empirical_kl_samples)
    tol = 10 * std(empirical_kl_samples) / sqrt(sample_size)
    min_diff = 1e-2 * std(empirical_kl_samples) / sqrt(sample_size)
    info("kl: $empirical_kl vs $claimed_kl [tol: $tol]")
    @test_approx_eq_eps empirical_kl claimed_kl tol
end


function verify_derivs(f::Function, x::Vector{Float64})
    claimed_dx = f(x)[2]

    for i in 1:length(x)
        f2(epsilon::Float64) = begin
            x_local = deepcopy(x)
            x_local[i] += epsilon
            f(x_local)[1]
        end

        numeric_deriv, abs_err = deriv_central(f2, 0., 1e-3)
        info("deriv #$i: $numeric_deriv vs $(claimed_dx[i]) [tol: $abs_err]")
        obs_err = abs(numeric_deriv - claimed_dx[i])
        @test obs_err < 1e-11 || abs_err < 1e-4 || abs_err / abs(numeric_deriv) < 1e-4
        @test_approx_eq_eps numeric_deriv claimed_dx[i] 10abs_err
    end
end


function test_beta()
    q = Beta(2., 5.)
    p = Beta(2., 5.)
    claimed_kl = KL.gen_beta_kl(2., 5.)(2., 5.)[1]
    verify_kl(q, p, claimed_kl)

    q = Beta(1., 1.)
    p = Beta(3., 3.)
    claimed_kl = KL.gen_beta_kl(3., 3.)(1., 1.)[1]
    verify_kl(q, p, claimed_kl)

    q = Beta(2., 5.)
    p = Beta(0.5, 0.5)
    claimed_kl = KL.gen_beta_kl(0.5, 0.5)(2., 5.)[1]
    verify_kl(q, p, claimed_kl)

    x = [2., 5.]
    f(x) = KL.gen_beta_kl(0.5, 0.5)(x[1], x[2])
    verify_derivs(f, x)
end


function test_wrappedcauchy_uniform()
    sample_size = 2_000_000
    q_scale = 0.3
    q_dist = Cauchy(0., q_scale)
    raw_q_samples = rand(q_dist, sample_size)
    wrap(x::Float64) = begin
        y::Float64 = (x % 2pi + 2pi) % 2pi
        (y < pi ? y : y - 2pi)::Float64
    end
    q_samples = Float64[wrap(x) for x in raw_q_samples]

    p_dist = Uniform(-pi, pi)
    q_logpdf(x) = log((sinh(q_scale) / (2pi * (cosh(q_scale) - cos(x)))))
    q_entropies = Float64[q_logpdf(x) for x in q_samples]
    empirical_kl_samples = q_entropies - logpdf(p_dist, q_samples)
    empirical_kl = mean(empirical_kl_samples)
    tol = 4 * std(empirical_kl_samples) / sqrt(sample_size)
    min_diff = 1e-2 * std(empirical_kl_samples) / sqrt(sample_size)

    claimed_kl = KL.gen_wrappedcauchy_uniform_kl()(q_scale)[1]
    @test_approx_eq_eps empirical_kl claimed_kl tol

    x = [0.3,]
    f(x) = KL.gen_wrappedcauchy_uniform_kl()(x[1])
    verify_derivs(f, x)
end


function test_categorical()
    q_probs = [0.1, 0.2, 0.7]
    p_probs = [0.3, 0.4, 0.3]
    q = Categorical(q_probs)
    p = Categorical(p_probs)

    claimed_kl = KL.gen_categorical_kl(p_probs)(q_probs)[1]
    verify_kl(q, p, claimed_kl)

    f(x) = begin
        v, dv = KL.gen_categorical_kl(p_probs)(x)
        v, dv[1]
    end
    verify_derivs(f, q_probs)
end


function test_gamma()
    q_k, q_theta = 3., 2.
    q = Gamma(q_k, q_theta)
    p_k, p_theta = 7.5, 1.
    p = Gamma(p_k, p_theta)

    claimed_kl = KL.gen_gamma_kl(p_k, p_theta)(q_k, q_theta)[1]
    verify_kl(q, p, claimed_kl)

    x = [q_k, q_theta]
    f(x) = KL.gen_gamma_kl(p_k, p_theta)(x[1], x[2])
    verify_derivs(f, x)
end


function test_univariate_normal()
    q = Normal(0., 1.)
    p = Normal(0., 1.)
    claimed_kl = (KL.gen_normal_kl(0., 1.))(0., 1.)[1]
    verify_kl(q, p, claimed_kl)

    q = Normal(0.3, sqrt(1.2))
    p = Normal(1.1, sqrt(2.12))
    claimed_kl = (KL.gen_normal_kl(1.1, 2.12))(0.3, 1.2)[1]
    verify_kl(q, p, claimed_kl)

    x = [0.3, 1.2]
    f(x) = KL.gen_normal_kl(1.1, 2.12)(x[1], x[2])
    verify_derivs(f, x)
end


function test_isobvnormal()
    q_mean = [1.2, 3.3]
    q = MvNormal(q_mean, sqrt(2.))
    p_mean = [2.1, 3.1]
    p = MvNormal(p_mean, sqrt(2.))
    claimed_kl = KL.gen_isobvnormal_kl(p_mean, 2.)(q_mean, 2.)[1]
    verify_kl(q, p, claimed_kl)

    q_mean = [1.2, 3.3]
    q = MvNormal(q_mean, sqrt(3.))
    p_mean = [2.1, 3.1]
    p = MvNormal(p_mean, sqrt(2.))
    claimed_kl = KL.gen_isobvnormal_kl(p_mean, 2.)(q_mean, 3.)[1]
    verify_kl(q, p, claimed_kl)

    f(x) = begin
        v, dv = KL.gen_isobvnormal_kl(p_mean, 2.)(x[1:2], x[3])
        v, [dv[1][1]; dv[1][2]; dv[2]]
    end
    verify_derivs(f, [q_mean; 3.])
end


function test_diagmvn_mvn()
    q_mean, q_vars = [1.,2, 3], [.1, 0.2, 1.]
    q = MvNormal(q_mean, sqrt(q_vars))
    p_mean, p_cov = q_mean, diagm(q_vars)
    p = MvNormal(p_mean, p_cov)
    claimed_kl = KL.gen_diagmvn_mvn_kl(p_mean, p_cov)(q_mean, q_vars)[1]
    verify_kl(q, p, claimed_kl)

    q_mean, q_vars = [1.,2, 3], [.1, 0.2, 1.]
    q = MvNormal(q_mean, sqrt(q_vars))
    A = randn(3,3)
    p_mean, p_cov = ones(3), A' * A
    p = MvNormal(p_mean, p_cov)
    claimed_kl = KL.gen_diagmvn_mvn_kl(p_mean, p_cov)(q_mean, q_vars)[1]
    verify_kl(q, p, claimed_kl)

    f(x) = begin
        v, dv = KL.gen_diagmvn_mvn_kl(p_mean, p_cov)(x[1:3], x[4:6])
        v, [dv[1]; dv[2]]
    end
    verify_derivs(f, [q_mean; q_vars])
end


function test_isobvnormal_flat()
    q_var = 0.1
    claimed_kl = KL.gen_isobvnormal_flat_kl()(q_var)[1]

    q = MvNormal([1.1, 2.2], sqrt(q_var))
    sample_size = 2_000_000
    q_samples = rand(q, sample_size)
    empirical_kl_samples = logpdf(q, q_samples)
    empirical_kl = mean(empirical_kl_samples)
    tol = 5 * std(empirical_kl_samples) / sqrt(sample_size)
    min_diff = 1e-2 * std(empirical_kl_samples) / sqrt(sample_size)
    info("kl: $empirical_kl vs $claimed_kl [tol: $tol]")
    @test_approx_eq_eps empirical_kl claimed_kl tol

    f(x)= KL.gen_isobvnormal_flat_kl()(x[1])
    verify_derivs(f, [q_var,])
end


test_beta()
test_wrappedcauchy_uniform()
test_categorical()
test_gamma()
test_univariate_normal()
test_isobvnormal()
test_diagmvn_mvn()
test_isobvnormal_flat()
