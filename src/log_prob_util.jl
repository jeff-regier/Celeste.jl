###################################################################
# Structs formerly from ..ElboDeriv that are used in both ELBO    #
# optimization and log_prob calculations                          #
###################################################################


"""
ElboArgs stores the arguments needed to evaluate the variational objective
function
"""
type ElboArgs{NumType <: Number}
    S::Int64
    N::Int64
    images::Vector{TiledImage}
    vp::VariationalParams{NumType}
    tile_source_map::Vector{Matrix{Vector{Int}}}
    patches::Matrix{SkyPatch}
    active_sources::Vector{Int}
end


function ElboArgs{NumType <: Number}(
            images::Vector{TiledImage},
            vp::VariationalParams{NumType},
            tile_source_map::Vector{Matrix{Vector{Int}}},
            patches::Matrix{SkyPatch},
            active_sources::Vector{Int})
    N = length(images)
    S = length(vp)
    @assert length(tile_source_map) == N
    @assert size(patches, 1) == S
    @assert size(patches, 2) == N
    ElboArgs(S, N, images, vp, tile_source_map, patches, active_sources)
end


# TODO: the identification of active pixels should go in pre-processing
type ActivePixel
    # image index
    n::Int

    # Linear tile index:
    tile_ind::Int

    # Location in tile:
    h::Int
    w::Int
end


"""
Get the active pixels (pixels for which the active sources are present).
TODO: move this to pre-processing and use it instead of setting low-signal
pixels to NaN.
"""
function get_active_pixels{NumType <: Number}(
                    ea::ElboArgs{NumType})
    active_pixels = ActivePixel[]

    for n in 1:ea.N, tile_ind in 1:length(ea.images[n].tiles)
        tile_sources = ea.tile_source_map[n][tile_ind]
        if length(intersect(tile_sources, ea.active_sources)) > 0
            tile = ea.images[n].tiles[tile_ind]
            h_width, w_width = size(tile.pixels)
            for w in 1:w_width, h in 1:h_width
                if !Base.isnan(tile.pixels[h, w])
                    push!(active_pixels, ActivePixel(n, tile_ind, h, w))
                end
            end
        end
    end

    active_pixels
end


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


"""
Populate fs0m_vec and fs1m_vec for all sources for a given pixel.

Args:
    - elbo_vars: Elbo intermediate values.
    - ea: Model parameters
    - tile_sources: A vector of integers of sources in 1:ea.S affecting the tile
    - tile: An ImageTile
    - h, w: The integer locations of the pixel within the tile
    - gal_mcs: Galaxy components
    - star_mcs: Star components

Returns:
    Updates elbo_vars.fs0m_vec and elbo_vars.fs1m_vec in place with the total
    shape contributions to this pixel's brightness.
"""
function populate_fsm_vecs!{NumType <: Number}(
                    elbo_vars::ElboIntermediateVariables{NumType},
                    ea::ElboArgs{NumType},
                    tile_sources::Vector{Int},
                    tile::ImageTile,
                    h::Int, w::Int,
                    gal_mcs::Array{GalaxyCacheComponent{NumType}, 4},
                    star_mcs::Array{BvnComponent{NumType}, 2})
    x = Float64[tile.h_range[h], tile.w_range[w]]
    for s in tile_sources
        # ensure tile.b is a filter band, not an image's index
        @assert 1 <= tile.b <= B
        wcs_jacobian = ea.patches[s, tile.b].wcs_jacobian
        active_source = s in ea.active_sources

        calculate_hessian =
            elbo_vars.calculate_hessian && elbo_vars.calculate_derivs && active_source
        clear!(elbo_vars.fs0m_vec[s], calculate_hessian)
        for k = 1:psf_K # PSF component
            accum_star_pos!(
                elbo_vars, s, star_mcs[k, s], x, wcs_jacobian, active_source)
        end

        clear!(elbo_vars.fs1m_vec[s], calculate_hessian)
        for i = 1:2 # Galaxy types
            for j in 1:8 # Galaxy component
                # If i == 2 then there are only six galaxy components.
                if (i == 1) || (j <= 6)
                    for k = 1:psf_K # PSF component
                        accum_galaxy_pos!(
                            elbo_vars, s, gal_mcs[k, j, i, s], x, wcs_jacobian,
                            active_source)
                    end
                end
            end
        end
    end
end


"""
Add the contributions of a star's bivariate normal term to the ELBO,
by updating elbo_vars.fs0m_vec[s] in place.

Args:
    - elbo_vars: Elbo intermediate values.
    - s: The index of the current source in 1:S
    - bmc: The component to be added
    - x: An offset for the component in pixel coordinates (e.g. a pixel location)
    - wcs_jacobian: The jacobian of the function pixel = F(world) at this location.
    - calculate_derivs: Whether to calculate derivatives.

Returns:
    Updates elbo_vars.fs0m_vec[s] in place.
"""
function accum_star_pos!{NumType <: Number}(
                    elbo_vars::ElboIntermediateVariables{NumType},
                    s::Int,
                    bmc::BvnComponent{NumType},
                    x::Vector{Float64},
                    wcs_jacobian::Array{Float64, 2},
                    calculate_derivs::Bool)
    eval_bvn_pdf!(elbo_vars.bvn_derivs, bmc, x)

    # TODO: Also make a version that doesn't calculate any derivatives
    # if the object isn't in active_sources.
    get_bvn_derivs!(elbo_vars.bvn_derivs, bmc, true, false)

    fs0m = elbo_vars.fs0m_vec[s]
    fs0m.v[1] += elbo_vars.bvn_derivs.f_pre[1]

    if elbo_vars.calculate_derivs && calculate_derivs
        transform_bvn_ux_derivs!(
            elbo_vars.bvn_derivs, wcs_jacobian, elbo_vars.calculate_hessian)
        bvn_u_d = elbo_vars.bvn_derivs.bvn_u_d
        bvn_uu_h = elbo_vars.bvn_derivs.bvn_uu_h

        # Accumulate the derivatives.
        for u_id in 1:2
            fs0m.d[star_ids.u[u_id]] += elbo_vars.bvn_derivs.f_pre[1] * bvn_u_d[u_id]
        end

        if elbo_vars.calculate_hessian
            # Hessian terms involving only the location parameters.
            # TODO: redundant term
            for u_id1 in 1:2, u_id2 in 1:2
                fs0m.h[star_ids.u[u_id1], star_ids.u[u_id2]] +=
                    elbo_vars.bvn_derivs.f_pre[1] * (bvn_uu_h[u_id1, u_id2] +
                    bvn_u_d[u_id1] * bvn_u_d[u_id2])
            end
        end
    end
end


"""
Add the contributions of a galaxy component term to the ELBO by
updating fs1m in place.

Args:
    - elbo_vars: Elbo intermediate variables
    - s: The index of the current source in 1:S
    - gcc: The galaxy component to be added
    - x: An offset for the component in pixel coordinates (e.g. a pixel location)
    - wcs_jacobian: The jacobian of the function pixel = F(world) at this location.
    - calculate_derivs: Whether to calculate derivatives.

Returns:
    Updates elbo_vars.fs1m_vec[s] in place.
"""
function accum_galaxy_pos!{NumType <: Number}(
                    elbo_vars::ElboIntermediateVariables{NumType},
                    s::Int,
                    gcc::GalaxyCacheComponent{NumType},
                    x::Vector{Float64},
                    wcs_jacobian::Array{Float64, 2},
                    calculate_derivs::Bool)
    eval_bvn_pdf!(elbo_vars.bvn_derivs, gcc.bmc, x)
    f = elbo_vars.bvn_derivs.f_pre[1] * gcc.e_dev_i
    fs1m = elbo_vars.fs1m_vec[s]
    fs1m.v[1] += f

    if elbo_vars.calculate_derivs && calculate_derivs

        get_bvn_derivs!(elbo_vars.bvn_derivs, gcc.bmc,
            elbo_vars.calculate_hessian, elbo_vars.calculate_hessian)
        transform_bvn_derivs!(
            elbo_vars.bvn_derivs, gcc.sig_sf, wcs_jacobian, elbo_vars.calculate_hessian)

        bvn_u_d = elbo_vars.bvn_derivs.bvn_u_d
        bvn_uu_h = elbo_vars.bvn_derivs.bvn_uu_h
        bvn_s_d = elbo_vars.bvn_derivs.bvn_s_d
        bvn_ss_h = elbo_vars.bvn_derivs.bvn_ss_h
        bvn_us_h = elbo_vars.bvn_derivs.bvn_us_h

        # Accumulate the derivatives.
        for u_id in 1:2
            fs1m.d[gal_ids.u[u_id]] += f * bvn_u_d[u_id]
        end

        for gal_id in 1:length(gal_shape_ids)
            fs1m.d[gal_shape_alignment[gal_id]] += f * bvn_s_d[gal_id]
        end

        # The e_dev derivative. e_dev just scales the entire component.
        # The direction is positive or negative depending on whether this
        # is an exp or dev component.
        fs1m.d[gal_ids.e_dev] += gcc.e_dev_dir * elbo_vars.bvn_derivs.f_pre[1]

        if elbo_vars.calculate_hessian
            # The Hessians:

            # Hessian terms involving only the shape parameters.
            for shape_id1 in 1:length(gal_shape_ids)
                for shape_id2 in 1:length(gal_shape_ids)
                    s1 = gal_shape_alignment[shape_id1]
                    s2 = gal_shape_alignment[shape_id2]
                    fs1m.h[s1, s2] +=
                        f * (bvn_ss_h[shape_id1, shape_id2] +
                                 bvn_s_d[shape_id1] * bvn_s_d[shape_id2])
                end
            end

            # Hessian terms involving only the location parameters.
            for u_id1 in 1:2, u_id2 in 1:2
                u1 = gal_ids.u[u_id1]
                u2 = gal_ids.u[u_id2]
                fs1m.h[u1, u2] +=
                    f * (bvn_uu_h[u_id1, u_id2] + bvn_u_d[u_id1] * bvn_u_d[u_id2])
            end

            # Hessian terms involving both the shape and location parameters.
            for u_id in 1:2, shape_id in 1:length(gal_shape_ids)
                ui = gal_ids.u[u_id]
                si = gal_shape_alignment[shape_id]
                fs1m.h[ui, si] +=
                    f * (bvn_us_h[u_id, shape_id] + bvn_u_d[u_id] * bvn_s_d[shape_id])
                fs1m.h[si, ui] = fs1m.h[ui, si]
            end

            # Do the e_dev hessian terms.
            devi = gal_ids.e_dev
            for u_id in 1:2
                ui = gal_ids.u[u_id]
                fs1m.h[ui, devi] +=
                    elbo_vars.bvn_derivs.f_pre[1] * gcc.e_dev_dir * bvn_u_d[u_id]
                fs1m.h[devi, ui] = fs1m.h[ui, devi]
            end
            for shape_id in 1:length(gal_shape_ids)
                si = gal_shape_alignment[shape_id]
                fs1m.h[si, devi] +=
                    elbo_vars.bvn_derivs.f_pre[1] * gcc.e_dev_dir * bvn_s_d[shape_id]
                fs1m.h[devi, si] = fs1m.h[si, devi]
            end
        end # if calculate hessian
    end # if calculate_derivs
end


"""
Convolve the current locations and galaxy shapes with the PSF.  If
calculate_derivs is true, also calculate derivatives and hessians for
active sources.

Args:
 - psf: A vector of PSF components
 - ea: The current ElboArgs
 - b: The current band
 - calculate_derivs: Whether to calculate derivatives for active sources.

Returns:
 - star_mcs: An array of BvnComponents with indices
    - PSF component
    - Source (index within active_sources)
 - gal_mcs: An array of BvnComponents with indices
    - PSF component
    - Galaxy component
    - Galaxy type
    - Source (index within active_sources)
  Hessians are only populated for s in ea.active_sources.
"""
function load_bvn_mixtures{NumType <: Number}(
    ea::ElboArgs{NumType}, b::Int;
    calculate_derivs::Bool=true, calculate_hessian::Bool=true)

  star_mcs = Array(BvnComponent{NumType}, psf_K, ea.S)
  gal_mcs = Array(GalaxyCacheComponent{NumType}, psf_K, 8, 2, ea.S)

  # TODO: do not keep any derviative information if the sources are not in
  # active_sources.
  for s in 1:ea.S
      psf = ea.patches[s, b].psf
      vs = ea.vp[s]

      world_loc = vs[[ids.u[1], ids.u[2]]]
      m_pos = linear_world_to_pix(ea.patches[s, b].wcs_jacobian,
                                  ea.patches[s, b].center,
                                  ea.patches[s, b].pixel_center,
                                  world_loc)

      # Convolve the star locations with the PSF.
      for k in 1:psf_K
          pc = psf[k]
          mean_s = [pc.xiBar[1] + m_pos[1], pc.xiBar[2] + m_pos[2]]
          star_mcs[k, s] =
            BvnComponent{NumType}(
              mean_s, pc.tauBar, pc.alphaBar, calculate_siginv_deriv=false)
      end

      # Convolve the galaxy representations with the PSF.
      for i = 1:2 # i indexes dev vs exp galaxy types.
          e_dev_dir = (i == 1) ? 1. : -1.
          e_dev_i = (i == 1) ? vs[ids.e_dev] : 1. - vs[ids.e_dev]

          # Galaxies of type 1 have 8 components, and type 2 have 6 components.
          for j in 1:[8,6][i]
              for k = 1:psf_K
                  gal_mcs[k, j, i, s] = GalaxyCacheComponent(
                      e_dev_dir, e_dev_i, galaxy_prototypes[i][j], psf[k],
                      m_pos, vs[ids.e_axis], vs[ids.e_angle], vs[ids.e_scale],
                      calculate_derivs && (s in ea.active_sources),
                      calculate_hessian)
              end
          end
      end
  end

  star_mcs, gal_mcs
end


