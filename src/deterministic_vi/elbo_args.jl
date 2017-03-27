"""
Store pre-allocated memory in this data structures, which contains
intermediate values used in the ELBO calculation.
"""
immutable HessianSubmatrices{NumType <: Number}
    u_u::Matrix{NumType}
    shape_shape::Matrix{NumType}
end


"""
Pre-allocated memory for efficiently accumulating certain sub-matrices
of the E_G_s and E_G2_s Hessian.

Args:
    NumType: The numeric type of the hessian.
    i: The type of celestial source, from 1:Ia
"""
function HessianSubmatrices(NumType::DataType, i::Int)
    @assert 1 <= i <= Ia
    shape_p = length(shape_standard_alignment[i])

    u_u = zeros(NumType, 2, 2)
    shape_shape = zeros(NumType, shape_p, shape_p)
    HessianSubmatrices{NumType}(u_u, shape_shape)
end

DenseHessianSSSF(ParamSet, NumType) = SingleSourceSensitiveFloat{NumType, ParamSet,
  ParameterizedArray{ParamSet, SizedMatrix{(length(ParamSet),length(ParamSet)),NumType,2}}}
SparseHessianCanonicalSSSF(NumType) = SingleSourceSensitiveFloat{NumType, CanonicalParams2, SparseStruct{NumType}}
DenseHessianSSparseSF(ParamSet, NumType) = SSparseSensitiveFloat{NumType, ParamSet,
  ParameterizedArray{ParamSet, SizedMatrix{(length(ParamSet),length(ParamSet)),NumType,2}}}
SparseHessianSSparseSF(ParamSet, NumType) = SSparseSensitiveFloat{NumType, ParamSet, SparseStruct{NumType}}
SensitiveFloats.zeros_type(T::Type{<:SparseStruct}, args...) = zeros(T)
immutable ElboIntermediateVariables{NumType <: Number}
    # Vectors of star and galaxy bvn quantities from all sources for a pixel.
    # The vector has one element for each active source, in the same order
    # as ea.active_sources.

    fs0m::DenseHessianSSSF(StarPosParams, NumType)
    fs1m::DenseHessianSSSF(GalaxyPosParams, NumType)

    # Brightness values for a single source
    E_G_s::SparseHessianCanonicalSSSF(NumType)
    E_G2_s::SparseHessianCanonicalSSSF(NumType)
    
    # Expected pixel intensity and variance for a pixel from all sources.
    E_G::SparseHessianSSparseSF(CanonicalParams2, NumType)
    var_G::DenseHessianSSparseSF(CanonicalParams2, NumType)

    # Pre-allocated memory for the gradient and Hessian of combine functions.
    combine_grad::Vector{NumType}
    combine_hess::Matrix{NumType}

    # The ELBO itself.
    elbo::SensitiveFloat{NumType, CanonicalParams2}

    active_pixel_counter::typeof(Ref{Int64}(0))
    inactive_pixel_counter::typeof(Ref{Int64}(0))
end


"""
Args:
    - num_active_sources: The number of actives sources (with derivatives)
    - calculate_gradient: If false, only calculate values
    - calculate_hessian: If false, only calculate gradients. Note that if
                calculate_gradient = false, then hessians will not be
                calculated irrespective of the value of calculate_hessian.
"""
function ElboIntermediateVariables(NumType::DataType,
                                   num_active_sources::Int,
                                   calculate_gradient::Bool=true,
                                   calculate_hessian::Bool=true)
    @assert NumType <: Number

    # fs0m and fs1m accumulate contributions from all bvn components
    # for a given source.
    fs0m = DenseHessianSSSF(StarPosParams, NumType)(
                                calculate_gradient, calculate_hessian)
    fs1m = DenseHessianSSSF(GalaxyPosParams, NumType)(
                                calculate_gradient, calculate_hessian)

    E_G_s = SparseHessianCanonicalSSSF(NumType)(
                                    calculate_gradient, calculate_hessian)
    E_G2_s = SensitiveFloat(E_G_s)

    E_G = SparseHessianSSparseSF(CanonicalParams2, NumType)(num_active_sources,
                                  calculate_gradient, calculate_hessian)
    var_G = DenseHessianSSparseSF(CanonicalParams2, NumType)(num_active_sources,
                                  calculate_gradient, calculate_hessian)

    combine_grad = zeros(NumType, 2)
    combine_hess = zeros(NumType, 2, 2)

    elbo = SensitiveFloat{NumType, CanonicalParams2}(num_active_sources,
                                    calculate_gradient, calculate_hessian)

    ElboIntermediateVariables{NumType}(
        fs0m, fs1m,
        E_G_s, E_G2_s,
        E_G, var_G, combine_grad, combine_hess,
        elbo, Ref{Int64}(0), Ref{Int64}(0))
end

function clear!{NumType <: Number}(elbo_vars::ElboIntermediateVariables{NumType})
    clear!(elbo_vars.fs0m)
    clear!(elbo_vars.fs1m)
    clear!(elbo_vars.E_G_s)
    clear!(elbo_vars.E_G2_s)

    clear!(elbo_vars.E_G)
    clear!(elbo_vars.var_G)

    fill!(elbo_vars.combine_grad, zero(NumType))
    fill!(elbo_vars.combine_hess, zero(NumType))

    clear!(elbo_vars.elbo)
end


"""
If Infs/NaNs have crept into the ELBO evaluation (a symptom of poorly conditioned optimization),
this helps catch them immediately.
"""
function assert_all_finite{NumType <: Number}(sf::SensitiveFloat{NumType})
    @assert isfinite(sf.v[]) "Value is Inf/NaNs"
    @assert all(isfinite, sf.d) "Gradient contains Inf/NaNs"
    @assert all(isfinite, sf.h) "Hessian contains Inf/NaNs"
end


"""
Some parameter to a function has invalid values. The message should explain what parameter is
invalid and why.
"""
type InvalidInputError <: Exception
    message::String
end


"""
ElboArgs stores the arguments needed to evaluate the variational objective
function.
"""
immutable ElboArgs
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
    images::Vector{Image}

    # subimages is a better name for patches: regions of an image
    # around a particular light source
    patches::Matrix{SkyPatch}

    # the sources to optimize
    active_sources::Vector{Int}

    # If false, elbo = elbo_likelihood
    include_kl::Bool
end


function ElboArgs(
            images::Vector{Image},
            patches::Matrix{SkyPatch},
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
