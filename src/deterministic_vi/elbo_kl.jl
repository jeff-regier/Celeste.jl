module KLDivergence

using ..DeterministicVI: ElboArgs
using ...Model: CanonicalParams, ids, prior, Ia, D
using ...SensitiveFloats: SensitiveFloat, add_sources_sf!
using ForwardDiff, ReverseDiff, DiffBase

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
    kl = log_term + apart_term + together_term
    assert(!isnan(kl))
    return kl
end

#############################
# Categorical Distributions #
#############################

"""
Returns the KL divergence between a pair of categorical distributions
"""
function categorical_kl(p₁, p₂)
    kl = sum(a * (log(a) - log(b)) for (a, b) in zip(p₁, p₂))
    assert(!isnan(kl))
    return kl
end

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
    assert(!isnan(kl))
    return 0.5 * kl
end

###########################################
# Subtracting KL divergences from sources #
###########################################

kl_source_a(vs) = categorical_kl(vs[ids.a], prior.a)

function kl_source_r(vs)
    kl = zero(eltype(vs))
    for i in 1:Ia
        kl += vs[ids.a[i]] * gaussian_kl(vs[ids.r1[i]], vs[ids.r2[i]],
                                            prior.r_μ[i], prior.r_σ²[i])
    end
    assert(!isnan(kl))
    return kl
end

function kl_source_k(vs)
    kl = zero(eltype(vs))
    for i in 1:Ia
        kl += vs[ids.a[i]] * categorical_kl(vs[ids.k[:, i]], prior.k[:, i])
    end
    assert(!isnan(kl))
    return kl
end

function kl_source_c(vs)
    kl = zero(eltype(vs))
    for i in 1:Ia
        μ₁, var₁ = vs[ids.c1[:, i]], vs[ids.c2[:, i]]
        a = vs[ids.a[i]]
        for d in 1:D
            μ₂, Σ₂ = prior.c_mean[:, d, i], prior.c_cov[:, :, d, i]
            kl += a * vs[ids.k[d, i]] * diagmvn_mvn_kl(μ₁, var₁, μ₂, Σ₂)
        end
    end
    assert(!isnan(kl))
    return kl
end


function source_e_log_prob(vs)
    x = vs[ids.e_scale]
    μ = prior.e_scale_μ
    σ² = prior.e_scale_σ²
    kl = -0.5 * (log(2pi) + log(σ²) + (x - μ)^2 / σ²)
    assert(!isnan(kl))
    return kl
end


"""
Subtract the KL divergences for a single source.
"""
function subtract_kl(vs)
    kl = zero(eltype(vs))
    kl -= kl_source_a(vs)
    kl -= kl_source_k(vs)
    kl -= kl_source_r(vs)
    kl -= kl_source_c(vs)

    # negative log probability is the kl divergence between a
    # variational distribution that is a point mass and the prior
    kl += source_e_log_prob(vs)
    return kl
end

###################
# Differentiation #
###################

using ForwardDiff: Dual, JacobianConfig, pickchunksize
using ReverseDiff: CompiledTape, GradientTape

immutable KLHelper{G,NG,D,J}
    gradient_tape::G
    nested_gradient_tape::NG
    dual_buffer::D
    jacobian_config::J
end

const PARAM_LENGTH = length(CanonicalParams)

function KLHelper{N,T}(::Type{Dual{N,T}})
    dual_buffer = zeros(Dual{N,T}, PARAM_LENGTH)
    jacobian_config = JacobianConfig{N}(rand(T, PARAM_LENGTH))
    gradient_tape = CompiledTape{:kl_gradient}(GradientTape(subtract_kl, rand(T, PARAM_LENGTH)))
    nested_gradient_tape = CompiledTape{:kl_nested_gradient}(GradientTape(subtract_kl, rand(Dual{N,T}, PARAM_LENGTH)))
    ReverseDiff.compile(gradient_tape)
    ReverseDiff.compile(nested_gradient_tape)
    return KLHelper(gradient_tape, nested_gradient_tape, dual_buffer, jacobian_config)
end

function kl_gradient!(out, x, helper::KLHelper)
    return ReverseDiff.gradient!(out, helper.gradient_tape, x)
end

function kl_hessian!(out, x, helper::KLHelper)
    f = x -> ReverseDiff.gradient!(helper.dual_buffer, helper.nested_gradient_tape, x)
    return ForwardDiff.jacobian!(out, f, x, helper.jacobian_config)
end


###############
# Entry Point #
###############

function subtract_kl_source!(kl_source::SensitiveFloat, result::DiffBase.DiffResult,
                             vs, helper::KLHelper)
    if kl_source.has_gradient
        kl_gradient!(result, vs, helper)
        kl_source.v[] = DiffBase.value(result)
    else
        kl_source.v[] = subtract_kl(vs)
    end
    if kl_source.has_hessian
        kl_hessian!(kl_source.h, vs, helper)
    end
    return kl_source
end

function subtract_kl_all_sources!{T}(ea::ElboArgs,
                                     accum::SensitiveFloat,
                                     kl_source::SensitiveFloat{T},
                                     helper::KLHelper)
    result = DiffBase.DiffResult(zero(T), kl_source.d)
    for sa in 1:length(ea.active_sources)
        subtract_kl_source!(kl_source, result, ea.vp[ea.active_sources[sa]], helper)
        add_sources_sf!(accum, kl_source, sa)
    end
    return accum
end

############
# __init__ #
############

const KL_HELPER_POOL_FLOAT = Vector{KLHelper}()
const KL_HELPER_POOL_DUAL = Vector{KLHelper}()

get_kl_helper(::Type{Float64}) = KL_HELPER_POOL_FLOAT[Base.Threads.threadid()]
get_kl_helper(::Type{Dual{1,Float64}}) = KL_HELPER_POOL_DUAL[Base.Threads.threadid()]

function __init__()
    N = pickchunksize(PARAM_LENGTH)
    DualFloat = Dual{N,Float64}
    DualDual = Dual{N,Dual{1,Float64}}
    if length(KL_HELPER_POOL_FLOAT) != Base.Threads.nthreads()
        empty!(KL_HELPER_POOL_FLOAT)
        empty!(KL_HELPER_POOL_DUAL)
        for i = 1:Base.Threads.nthreads()
            push!(KL_HELPER_POOL_FLOAT, KLHelper(DualFloat))
            push!(KL_HELPER_POOL_DUAL, KLHelper(DualDual))
        end
    end
end

# explicitly call this for use with compiled system image
__init__()

end # module
