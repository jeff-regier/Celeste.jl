"""
Store pre-allocated memory in this data structures, which contains
intermediate values used in the ELBO calculation.
"""
struct HessianSubmatrices{T<:Number}
    u_u::Matrix{T}
    shape_shape::Matrix{T}
end


"""
Pre-allocated memory for efficiently accumulating certain sub-matrices
of the E_G_s and E_G2_s Hessian.

Args:
    T: The numeric type of the hessian.
    i: The type of celestial source, from 1:NUM_SOURCE_TYPES
"""
function HessianSubmatrices(::Type{T}, i::Int) where {T}
    @assert 1 <= i <= NUM_SOURCE_TYPES
    shape_p = length(shape_standard_alignment[i])

    u_u = zeros(T, 2, 2)
    shape_shape = zeros(T, shape_p, shape_p)
    HessianSubmatrices{T}(u_u, shape_shape)
end


struct ElboIntermediateVariables{T<:Number}
    # Vectors of star and galaxy bvn quantities from all sources for a pixel.
    # The vector has one element for each active source, in the same order
    # as ea.active_sources.

    fs0m::SensitiveFloat{T}
    fs1m::SensitiveFloat{T}

    # Brightness values for a single source
    E_G_s::SensitiveFloat{T}
    E_G2_s::SensitiveFloat{T}
    var_G_s::SensitiveFloat{T}

    # Subsets of the Hessian of E_G_s and E_G2_s that allow us to use BLAS
    # functions to accumulate Hessian terms. There is one submatrix for
    # each celestial object type in 1:NUM_SOURCE_TYPES
    E_G_s_hsub_vec::Vector{HessianSubmatrices{T}}
    E_G2_s_hsub_vec::Vector{HessianSubmatrices{T}}

    # Expected pixel intensity and variance for a pixel from all sources.
    E_G::SensitiveFloat{T}
    var_G::SensitiveFloat{T}

    # Pre-allocated memory for the gradient and Hessian of combine functions.
    combine_grad::Vector{T}
    combine_hess::Matrix{T}

    # A placeholder for the log term in the ELBO.
    elbo_log_term::SensitiveFloat{T}

    # The ELBO itself.
    elbo::SensitiveFloat{T}

    active_pixel_counter::Ref{Int64}
    inactive_pixel_counter::Ref{Int64}
end


"""
Args:
    - num_active_sources: The number of actives sources (with derivatives)
    - calculate_gradient: If false, only calculate values
    - calculate_hessian: If false, only calculate gradients. Note that if
                calculate_gradient = false, then hessians will not be
                calculated irrespective of the value of calculate_hessian.
"""
function ElboIntermediateVariables(
        ::Type{T},
        num_active_sources::Int,
        calculate_gradient::Bool=true,
        calculate_hessian::Bool=true) where {T<:Number}

    # fs0m and fs1m accumulate contributions from all bvn components
    # for a given source.
    fs0m = SensitiveFloat{T}(length(StarPosParams), 1,
                             calculate_gradient, calculate_hessian)
    fs1m = SensitiveFloat{T}(length(GalaxyPosParams), 1,
                             calculate_gradient, calculate_hessian)

    E_G_s = SensitiveFloat{T}(length(CanonicalParams), 1,
                              calculate_gradient, calculate_hessian)
    E_G2_s = SensitiveFloat(E_G_s)
    var_G_s = SensitiveFloat(E_G_s)

    E_G_s_hsub_vec =
        HessianSubmatrices{T}[ HessianSubmatrices(T, i) for i=1:NUM_SOURCE_TYPES ]
    E_G2_s_hsub_vec =
        HessianSubmatrices{T}[ HessianSubmatrices(T, i) for i=1:NUM_SOURCE_TYPES ]

    E_G = SensitiveFloat{T}(length(CanonicalParams), num_active_sources,
                            calculate_gradient, calculate_hessian)
    var_G = SensitiveFloat(E_G)

    combine_grad = zeros(T, 2)
    combine_hess = zeros(T, 2, 2)

    elbo_log_term = SensitiveFloat(E_G)
    elbo = SensitiveFloat(E_G)

    ElboIntermediateVariables{T}(
        fs0m, fs1m,
        E_G_s, E_G2_s, var_G_s, E_G_s_hsub_vec, E_G2_s_hsub_vec,
        E_G, var_G, combine_grad, combine_hess,
        elbo_log_term, elbo, 0, 0)
end


function zero!(elbo_vars::ElboIntermediateVariables{T}) where {T<:Number}
    SensitiveFloats.zero!(elbo_vars.fs0m)
    SensitiveFloats.zero!(elbo_vars.fs1m)
    SensitiveFloats.zero!(elbo_vars.E_G_s)
    SensitiveFloats.zero!(elbo_vars.E_G2_s)
    SensitiveFloats.zero!(elbo_vars.var_G_s)

    for i in 1:NUM_SOURCE_TYPES
        fill!(elbo_vars.E_G_s_hsub_vec[i].u_u, zero(T))
        fill!(elbo_vars.E_G_s_hsub_vec[i].shape_shape, zero(T))
        fill!(elbo_vars.E_G2_s_hsub_vec[i].u_u, zero(T))
        fill!(elbo_vars.E_G2_s_hsub_vec[i].shape_shape, zero(T))
    end

    SensitiveFloats.zero!(elbo_vars.E_G)
    SensitiveFloats.zero!(elbo_vars.var_G)

    fill!(elbo_vars.combine_grad, zero(T))
    fill!(elbo_vars.combine_hess, zero(T))

    SensitiveFloats.zero!(elbo_vars.elbo_log_term)
    SensitiveFloats.zero!(elbo_vars.elbo)
end


"""
If Infs/NaNs have crept into the ELBO evaluation (a symptom of poorly conditioned optimization),
this helps catch them immediately.
"""
function assert_all_finite(sf::SensitiveFloat{T}) where {T<:Number}
    @assert isfinite(sf.v[]) "Value is Inf/NaNs"
    @assert all(isfinite, sf.d) "Gradient contains Inf/NaNs"
    @assert all(isfinite, sf.h) "Hessian contains Inf/NaNs"
end


"""
Some parameter to a function has invalid values. The message should explain what parameter is
invalid and why.
"""
struct InvalidInputError <: Exception
    message::String
end


"""
ElboArgs stores the arguments needed to evaluate the variational objective
function.
"""
struct ElboArgs
    # the overall number of sources: we don't necessarily visit them
    # all or optimize them all, but if we do visit a pixel where any
    # of these are active, we use it in the elbo calculation
    S::Int64

    # number of active sources (see below)
    Sa::Int64

    # the number of images
    N::Int64

    # The number of components in the point spread function.
    psf_K::Int64
    images::Vector{<:Image}

    # subimages is a better name for patches: regions of an image
    # around a particular light source
    patches::Matrix{ImagePatch}

    # the sources to optimize
    active_sources::Vector{Int}

    # If false, elbo = elbo_likelihood
    include_kl::Bool
end


function ElboArgs(
            images::Vector{<:Image},
            patches::Matrix{ImagePatch},
            active_sources::Vector{Int};
            psf_K::Int=2,
            include_kl::Bool=true)
    S = size(patches, 1)
    Sa = length(active_sources)
    N = length(images)

    @assert size(patches, 2) == N
    @assert psf_K > 0

    @assert(length(active_sources) <= 5 || !calculate_hessian,
            "too many active_sources to store a hessian")

    ElboArgs(S, Sa, N, psf_K, images, patches,
             active_sources, include_kl)
end
