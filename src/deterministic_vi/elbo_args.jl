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

DenseHessianSSSF(ParamSet, NumType, HasGradient, HasHessian) = SingleSourceSensitiveFloat{NumType, ParamSet,
  ParameterizedArray{ParamSet, SizedMatrix{(length(ParamSet),length(ParamSet)),NumType,2}}, HasGradient, HasHessian}
SparseHessianCanonicalSSSF(NumType, HasGradient, HasHessian) = SingleSourceSensitiveFloat{NumType, CanonicalParams2, SparseStruct{NumType}, HasGradient, HasHessian}
SparseHessianGalPosSSSF(NumType, HasGradient, HasHessian) = SingleSourceSensitiveFloat{NumType, GalaxyPosParams, SparseGalPosParams{NumType}, HasGradient, HasHessian}
DenseHessianSSparseSF(ParamSet, NumType, HasGradient, HasHessian) = SSparseSensitiveFloat{NumType, ParamSet,
  ParameterizedArray{ParamSet, SizedMatrix{(length(ParamSet),length(ParamSet)),NumType,2}}, HasGradient, HasHessian}
DenseSymmetricHessianSSparseSF(ParamSet, NumType) = SSparseSensitiveFloat{NumType, ParamSet,
  ParameterizedArray{ParamSet, Symmetric2{NumType, SizedVector{
    div(length(ParamSet)*(length(ParamSet)+1),2), NumType, 1
  }, :U}}}
SparseHessianSSparseSF(ParamSet, NumType, HasGradient, HasHessian) = SSparseSensitiveFloat{NumType, ParamSet, SparseStruct{NumType}, HasGradient, HasHessian}
SensitiveFloats.zeros_type(T::Type{<:SparseStruct}, args...) = zeros(T)
SensitiveFloats.zeros_type(T::Type{<:SparseGalPosParams}, args...) = zeros(T)

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
