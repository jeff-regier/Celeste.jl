immutable ElboIntermediateVariables{NumType <: Number, ElboRep}
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

    # The ELBO itself in whatever representation is most efficient inside the loop
    elbo::ElboRep
    
    # The ELBO after a once-per-evaluation reparameterization step
    reparameterized_elbo::SensitiveFloat{NumType, CanonicalParams, Matrix{NumType}}

    # Some scratch space for propagate_derivatives!
    pd_scratch::Matrix{NumType}

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

    elbo = SensitiveFloat{NumType, CanonicalParams2,
      SizedMatrix{(length(CanonicalParams2)*num_active_sources,length(CanonicalParams2)*num_active_sources),NumType,2}}(
        num_active_sources, calculate_gradient, calculate_hessian)
        
    reparameterized_elbo = SensitiveFloat{NumType, CanonicalParams, Matrix{Float64}}(
      num_active_sources, calculate_gradient, calculate_hessian)

    pd_scratch = Matrix{NumType}(length(CanonicalParams), length(CanonicalParams))

    ElboIntermediateVariables{NumType, typeof(elbo)}(
        fs0m, fs1m,
        E_G_s, E_G2_s,
        E_G, var_G,
        elbo, reparameterized_elbo, pd_scratch, Ref{Int64}(0), Ref{Int64}(0))
end

function clear!{NumType <: Number}(elbo_vars::ElboIntermediateVariables{NumType})
    clear!(elbo_vars.fs0m)
    clear!(elbo_vars.fs1m)
    clear!(elbo_vars.E_G_s)
    clear!(elbo_vars.E_G2_s)

    clear!(elbo_vars.E_G)
    clear!(elbo_vars.var_G)

    clear!(elbo_vars.elbo)
end

immutable BvnBundle{T<:Real}
    bvn_derivs::BivariateNormalDerivatives{T}
    star_mcs::Matrix{BvnComponent{T}}
    gal_mcs::Array{GalaxyCacheComponent{T},4}
    sbs::Vector{SourceBrightness{T}}
    function (::Type{BvnBundle{T}}){T}(psf_K::Int, S::Int, calculate_gradient::Bool=true, calculate_hessian::Bool=true)
        return new{T}(BivariateNormalDerivatives{T}(),
                      Matrix{BvnComponent{T}}(psf_K, S),
                      Array{GalaxyCacheComponent{T}}(psf_K, 8, 2, S),
                      [SourceBrightness{T}(calculate_gradient, calculate_hessian) for i = 1:S])
    end
end

clear!(bvn_bundle::BvnBundle) = (clear!(bvn_bundle.bvn_derivs); bvn_bundle)
