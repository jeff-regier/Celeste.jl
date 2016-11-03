"""
Store pre-allocated memory in this data structures, which contains
intermediate values used in the ELBO calculation.
"""
type HessianSubmatrices{NumType <: Number}
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


type ElboIntermediateVariables{NumType <: Number}

    bvn_derivs::BivariateNormalDerivatives{NumType}

    # Vectors of star and galaxy bvn quantities from all sources for a pixel.
    # The vector has one element for each active source, in the same order
    # as ea.active_sources.

    # TODO: you can treat this the same way as E_G_s and not keep a vector around.
    fs0m_vec::Vector{SensitiveFloat{StarPosParams, NumType}}
    fs1m_vec::Vector{SensitiveFloat{GalaxyPosParams, NumType}}

    # Brightness values for a single source
    E_G_s::SensitiveFloat{CanonicalParams, NumType}
    E_G2_s::SensitiveFloat{CanonicalParams, NumType}
    var_G_s::SensitiveFloat{CanonicalParams, NumType}

    # Subsets of the Hessian of E_G_s and E_G2_s that allow us to use BLAS
    # functions to accumulate Hessian terms. There is one submatrix for
    # each celestial object type in 1:Ia
    E_G_s_hsub_vec::Vector{HessianSubmatrices{NumType}}
    E_G2_s_hsub_vec::Vector{HessianSubmatrices{NumType}}

    # Expected pixel intensity and variance for a pixel from all sources.
    E_G::SensitiveFloat{CanonicalParams, NumType}
    var_G::SensitiveFloat{CanonicalParams, NumType}

    # Pre-allocated memory for the gradient and Hessian of combine functions.
    combine_grad::Vector{NumType}
    combine_hess::Matrix{NumType}

    # A placeholder for the log term in the ELBO.
    elbo_log_term::SensitiveFloat{CanonicalParams, NumType}

    # The ELBO itself.
    elbo::SensitiveFloat{CanonicalParams, NumType}

    # If false, do not calculate hessians or derivatives.
    calculate_derivs::Bool

    # If false, do not calculate hessians.
    calculate_hessian::Bool
end


"""
Args:
    - S: The total number of sources
    - num_active_sources: The number of actives sources (with deriviatives)
    - calculate_derivs: If false, only calculate values
    - calculate_hessian: If false, only calculate gradients. Note that if
                calculate_derivs = false, then hessians will not be
                calculated irrespective of the value of calculate_hessian.
"""
function ElboIntermediateVariables(NumType::DataType,
                                   S::Int,
                                   num_active_sources::Int;
                                   calculate_derivs::Bool=true,
                                   calculate_hessian::Bool=true)
    @assert NumType <: Number

    bvn_derivs = BivariateNormalDerivatives{NumType}(NumType)

    # fs0m and fs1m accumulate contributions from all bvn components
    # for a given source.
    fs0m_vec = Array(SensitiveFloat{StarPosParams, NumType}, S)
    fs1m_vec = Array(SensitiveFloat{GalaxyPosParams, NumType}, S)
    for s = 1:S
        fs0m_vec[s] = zero_sensitive_float(StarPosParams, NumType)
        fs1m_vec[s] = zero_sensitive_float(GalaxyPosParams, NumType)
    end

    E_G_s = zero_sensitive_float(CanonicalParams, NumType, 1)
    E_G2_s = zero_sensitive_float(CanonicalParams, NumType, 1)
    var_G_s = zero_sensitive_float(CanonicalParams, NumType, 1)

    E_G_s_hsub_vec =
        HessianSubmatrices{NumType}[ HessianSubmatrices(NumType, i) for i=1:Ia ]
    E_G2_s_hsub_vec =
        HessianSubmatrices{NumType}[ HessianSubmatrices(NumType, i) for i=1:Ia ]

    E_G = zero_sensitive_float(CanonicalParams, NumType, num_active_sources)
    var_G = zero_sensitive_float(CanonicalParams, NumType, num_active_sources)

    combine_grad = zeros(NumType, 2)
    combine_hess = zeros(NumType, 2, 2)

    elbo_log_term =
        zero_sensitive_float(CanonicalParams, NumType, num_active_sources)
    elbo = zero_sensitive_float(CanonicalParams, NumType, num_active_sources)

    ElboIntermediateVariables{NumType}(
        bvn_derivs, fs0m_vec, fs1m_vec,
        E_G_s, E_G2_s, var_G_s, E_G_s_hsub_vec, E_G2_s_hsub_vec,
        E_G, var_G, combine_grad, combine_hess,
        elbo_log_term, elbo, calculate_derivs, calculate_hessian)
end


function clear!{NumType <: Number}(elbo_vars::ElboIntermediateVariables{NumType})
    #TODO: don't allocate memory here?
    elbo_vars.bvn_derivs = BivariateNormalDerivatives{NumType}(NumType)

    for s = 1:length(elbo_vars.fs0m_vec)
        clear!(elbo_vars.fs0m_vec[s])
        clear!(elbo_vars.fs1m_vec[s])
    end

    clear!(elbo_vars.E_G_s)
    clear!(elbo_vars.E_G2_s)
    clear!(elbo_vars.var_G_s)

    for i in 1:Ia
        fill!(elbo_vars.E_G_s_hsub_vec[i].u_u, zero(NumType))
        fill!(elbo_vars.E_G_s_hsub_vec[i].shape_shape, zero(NumType))
        fill!(elbo_vars.E_G2_s_hsub_vec[i].u_u, zero(NumType))
        fill!(elbo_vars.E_G2_s_hsub_vec[i].shape_shape, zero(NumType))
    end

    clear!(elbo_vars.E_G)
    clear!(elbo_vars.var_G)

    fill!(elbo_vars.combine_grad, zero(NumType))
    fill!(elbo_vars.combine_hess, zero(NumType))

    clear!(elbo_vars.elbo_log_term)
    clear!(elbo_vars.elbo)
end


"""
If Infs/NaNs have crept into the ELBO evaluation (a symptom of poorly conditioned optimization),
this helps catch them immediately.
"""
function assert_all_finite{ParamType <: ParamSet, NumType <: Number}(
        sf::SensitiveFloat{ParamType, NumType})
    @assert all(isfinite, sf.v) "Value is Inf/NaNs"
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
type ElboArgs{NumType <: Number}
    S::Int64
    N::Int64

    # The number of components in the point spread function.
    psf_K::Int64
    images::Vector{TiledImage}
    vp::VariationalParams{NumType}
    tile_source_map::Vector{Matrix{Vector{Int}}}
    patches::Matrix{SkyPatch}
    active_sources::Vector{Int}

    # Bivarite normals will not be evaulated at points further than this many
    # standard deviations away from their mean.  See its usage in the ELBO and
    # bivariate normals for details.
    #
    # If this is set to Inf, the bivariate normals will be evaluated at all points
    # irrespective of their distance from the mean.
    num_allowed_sd::Float64

    active_pixels::Vector{ActivePixel}

    elbo_vars::ElboIntermediateVariables
end


function ElboArgs{NumType <: Number}(
            images::Vector{TiledImage},
            vp::VariationalParams{NumType},
            tile_source_map::Vector{Matrix{Vector{Int}}},
            patches::Matrix{SkyPatch},
            active_sources::Vector{Int};
            psf_K::Int=2,
            num_allowed_sd::Float64=Inf)
    N = length(images)
    S = length(vp)

    @assert psf_K > 0
    @assert length(tile_source_map) == N
    @assert size(patches, 1) == S
    @assert size(patches, 2) == N

    for img in images, tile in img.tiles, ep in tile.epsilon_mat
        if ep <= 0.0
            throw(InvalidInputError(
                "You must set all values of epsilon_mat > 0 for all images included in ElboArgs"
            ))
        end
    end

    @assert(length(active_sources) <= 5, "too many active_sources")

    elbo_vars = ElboIntermediateVariables(NumType, S,
                                length(active_sources),
                                calculate_derivs=true,
                                calculate_hessian=true)

    ElboArgs(S, N, psf_K, images, vp, tile_source_map, patches,
             active_sources, num_allowed_sd, ActivePixel[], elbo_vars)
end

