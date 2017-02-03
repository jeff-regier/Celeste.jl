using Celeste: Model, SensitiveFloats, DeterministicVI
using DeterministicVI: KLDivergence
using Distributions, DiffBase, JLD
using Base.Test

"""
Use Monte Carlo to check whether KL(q_dist || p_dist) matches exact_kl

Args:
    q_dist, p_dist: Distribution objects
    exact_kl: The expected exact KL
"""
function test_kl_value(q_dist, p_dist, exact_kl::Float64)
    sample_size = 2_000_000
    q_samples = rand(q_dist, sample_size)
    empirical_kl_samples = logpdf(q_dist, q_samples) - logpdf(p_dist, q_samples)
    empirical_kl = mean(empirical_kl_samples)
    tol = 4 * std(empirical_kl_samples) / sqrt(sample_size)
    min_diff = 1e-2 * std(empirical_kl_samples) / sqrt(sample_size)
    @test isapprox(empirical_kl, exact_kl, atol=tol)
end

function test_beta_kl_value()
    alpha1, beta1 = 4.1, 3.9
    alpha2, beta2 = 3.5, 4.3
    p1_dist = Beta(alpha1, beta1)
    p2_dist = Beta(alpha2, beta2)
    kl = KLDivergence.beta_kl(alpha1, beta1, alpha2, beta2)
    test_kl_value(p1_dist, p2_dist, kl)
end

function test_categorical_kl_value()
    p1 = Float64[1, 2, 3, 4]
    p2 = Float64[5, 6, 2, 1]
    p1 = p1 ./ sum(p1)
    p2 = p2 ./ sum(p2)
    p1_dist = Categorical(p1)
    p2_dist = Categorical(p2)
    kl = KLDivergence.categorical_kl(p1, p2)
    test_kl_value(p1_dist, p2_dist, kl)
end

function test_diagmvn_mvn_kl_value()
    K = 4
    mean1 = rand(K)
    var1 = rand(K)
    var1 = var1 .* var1
    mean2 = rand(K)
    cov2 = rand(K, K)
    cov2 = 0.2 * cov2 * cov2' + eye(K)
    cov2 = 0.5 * (cov2 + cov2')
    p1_dist = MvNormal(mean1, diagm(var1))
    p2_dist = MvNormal(mean2, cov2)
    kl = KLDivergence.diagmvn_mvn_kl(mean1, var1, mean2, cov2)
    test_kl_value(p1_dist, p2_dist, kl)
end

function test_gaussian_kl_value()
    mean1, var1 = 0.5, 2.0
    mean2, var2 = 0.8, 1.8
    p1_dist = Normal(mean1, sqrt(var1))
    p2_dist = Normal(mean2, sqrt(var2))
    kl = KLDivergence.gaussian_kl(mean1, var1, mean2, var2)
    test_kl_value(p1_dist, p2_dist, kl)
end

function test_subtract_kl()
    sf = SensitiveFloat{Float64}(32, 1, true, true)
    vs = rand(MersenneTwister(1), 32)
    kl_result = DiffBase.DiffResult(0.0, sf.d)
    kl_helper = KLDivergence.get_kl_helper(Float64)
    KLDivergence.subtract_kl_source!(sf, kl_result, vs, kl_helper)
    @test sf.v[] == DiffBase.value(kl_result)
    @test sf.d === DiffBase.gradient(kl_result)
    test_sf = JLD.load(joinpath(datadir, "kl_values.jld"), "sf")
    @test sf.v[] ≈ test_sf.v[]
    @test sf.d   ≈ test_sf.d
    @test sf.h   ≈ test_sf.h
end

println("Running KL divergence tests.")
test_beta_kl_value()
test_categorical_kl_value()
test_diagmvn_mvn_kl_value()
test_gaussian_kl_value()
@test_skip test_subtract_kl()
