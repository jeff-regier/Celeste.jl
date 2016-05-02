using Distributions

using SensitiveFloats
include("../src/elbo_kl.jl")


function verify_kl(q_dist, p_dist, claimed_kl::Float64)
    sample_size = 4_000_000
    q_samples = rand(q_dist, sample_size)
    empirical_kl_samples = logpdf(q_dist, q_samples) - logpdf(p_dist, q_samples)
    empirical_kl = mean(empirical_kl_samples)
    tol = max(10 * std(empirical_kl_samples) / sqrt(sample_size), 1e-12)
    min_diff = 1e-2 * std(empirical_kl_samples) / sqrt(sample_size)
    info("kl: $empirical_kl vs $claimed_kl [tol: $tol]")
    @test_approx_eq_eps empirical_kl claimed_kl tol
end


function test_categorical()
    q_probs = [0.1, 0.2, 0.7]
    p_probs = [0.3, 0.4, 0.3]
    q = Categorical(q_probs)
    p = Categorical(p_probs)

    claimed_kl = gen_categorical_kl(p_probs)(q_probs)[1]
    verify_kl(q, p, claimed_kl)
end


function test_univariate_normal()
    q = Normal(0., 1.)
    p = Normal(0., 1.)
    claimed_kl = (gen_normal_kl(0., 1.))(0., 1.)[1]
    verify_kl(q, p, claimed_kl)

    q = Normal(0.3, sqrt(1.2))
    p = Normal(1.1, sqrt(2.12))
    claimed_kl = (gen_normal_kl(1.1, 2.12))(0.3, 1.2)[1]
    verify_kl(q, p, claimed_kl)
end


function test_diagmvn_mvn()
    q_mean, q_vars = [1.,2, 3], [.1, 0.2, 1.]
    q = MvNormal(q_mean, sqrt(q_vars))
    p_mean, p_cov = q_mean, diagm(q_vars)
    p = MvNormal(p_mean, p_cov)
    claimed_kl = gen_diagmvn_mvn_kl(p_mean, p_cov)(q_mean, q_vars)[1]
    verify_kl(q, p, claimed_kl)

    q_mean, q_vars = [1.,2, 3], [.1, 0.2, 1.]
    q = MvNormal(q_mean, sqrt(q_vars))
    A = randn(3,3)
    p_mean, p_cov = ones(3), A' * A
    p = MvNormal(p_mean, p_cov)
    claimed_kl = gen_diagmvn_mvn_kl(p_mean, p_cov)(q_mean, q_vars)[1]
    verify_kl(q, p, claimed_kl)
end


test_categorical()
test_univariate_normal()
test_diagmvn_mvn()
