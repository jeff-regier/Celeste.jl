"""
Calculate values and partial derivatives of the variational ELBO.
"""
module ElboDeriv

import ForwardDiff

using ..Model
using ..SensitiveFloats
using ..BivariateNormals

import ..KL
import ..WCSUtils

include("elbo_kl.jl")
include("source_brightness.jl")


# TODO: the identification of active pixels should go in pre-processing
type ActivePixel
    # Band:
    b::Int

    # Linear tile index:
    tile_ind::Int

    # Location in tile:
    h::Int
    w::Int
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

    u_u = zeros(NumType, 2, 2);
    shape_shape = zeros(NumType, shape_p, shape_p);
    HessianSubmatrices{NumType}(u_u, shape_shape)
end


type ElboIntermediateVariables{NumType <: Number}

    bvn_derivs::BivariateNormalDerivatives{NumType}

    # Vectors of star and galaxy bvn quantities from all sources for a pixel.
    # The vector has one element for each active source, in the same order
    # as mp.active_sources.

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
    get_bvn_derivs!(elbo_vars.bvn_derivs, bmc, true, false);

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

    true # Set return type
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
    fs1m = elbo_vars.fs1m_vec[s];
    fs1m.v[1] += f

    if elbo_vars.calculate_derivs && calculate_derivs

        get_bvn_derivs!(elbo_vars.bvn_derivs, gcc.bmc,
            elbo_vars.calculate_hessian, elbo_vars.calculate_hessian);
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
            for shape_id1 in 1:length(gal_shape_ids), shape_id2 in 1:length(gal_shape_ids)
                s1 = gal_shape_alignment[shape_id1]
                s2 = gal_shape_alignment[shape_id2]
                fs1m.h[s1, s2] +=
                    f * (bvn_ss_h[shape_id1, shape_id2] +
                             bvn_s_d[shape_id1] * bvn_s_d[shape_id2])
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

    true # Set return type
end


"""
Populate fs0m_vec and fs1m_vec for all sources for a given pixel.

Args:
    - elbo_vars: Elbo intermediate values.
    - mp: Model parameters
    - tile_sources: A vector of integers of sources in 1:mp.S affecting the tile
    - tile: An ImageTile
    - h, w: The integer locations of the pixel within the tile
    - sbs: Source brightnesses
    - gal_mcs: Galaxy components
    - star_mcs: Star components

Returns:
    Updates elbo_vars.fs0m_vec and elbo_vars.fs1m_vec in place with the total
    shape contributions to this pixel's brightness.
"""
function populate_fsm_vecs!{NumType <: Number}(
                    elbo_vars::ElboIntermediateVariables{NumType},
                    mp::ModelParams{NumType},
                    tile_sources::Vector{Int},
                    tile::ImageTile,
                    h::Int, w::Int,
                    sbs::Vector{SourceBrightness{NumType}},
                    gal_mcs::Array{GalaxyCacheComponent{NumType}, 4},
                    star_mcs::Array{BvnComponent{NumType}, 2})
    x = Float64[tile.h_range[h], tile.w_range[w]]
    for s in tile_sources
        wcs_jacobian = mp.patches[s, tile.b].wcs_jacobian;
        active_source = s in mp.active_sources

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
Add the contributions of a single source to E_G_s and var_G_s, which are cleared
and then updated in place.

Args:
    - elbo_vars: Elbo intermediate values, with updated fs1m_vec and fs0m_vec.
    - mp: Model parameters
    - sbs: Source brightnesses
    - s: The source, in 1:mp.S
    - b: The band

Returns:
    Updates elbo_vars.E_G_s and elbo_vars.var_G_s in place with the brightness
    for this sourve at this pixel.
"""
function accumulate_source_brightness!{NumType <: Number}(
                    elbo_vars::ElboIntermediateVariables{NumType},
                    mp::ModelParams{NumType},
                    sbs::Vector{SourceBrightness{NumType}},
                    s::Int, b::Int)
    # E[G] and E{G ^ 2} for a single source
    E_G_s = elbo_vars.E_G_s;
    E_G2_s = elbo_vars.E_G2_s;

    clear_hessian = elbo_vars.calculate_hessian && elbo_vars.calculate_derivs
    clear!(E_G_s, clear_hessian)
    clear!(E_G2_s, clear_hessian)

    sb = sbs[s];

    active_source = (s in mp.active_sources)

    for i in 1:Ia # Stars and galaxies
        fsm_i = (i == 1) ? elbo_vars.fs0m_vec[s] : elbo_vars.fs1m_vec[s]
        a_i = mp.vp[s][ids.a[i]]

        lf = sb.E_l_a[b, i].v[1] * fsm_i.v[1]
        llff = sb.E_ll_a[b, i].v[1] * fsm_i.v[1]^2

        E_G_s.v[1] += a_i * lf
        E_G2_s.v[1] += a_i * llff

        # Only calculate derivatives for active sources.
        if active_source && elbo_vars.calculate_derivs
            ######################
            # Gradients.

            E_G_s.d[ids.a[i], 1] += lf
            E_G2_s.d[ids.a[i], 1] += llff

            p0_shape = shape_standard_alignment[i]
            p0_bright = brightness_standard_alignment[i]
            u_ind = i == 1 ? star_ids.u : gal_ids.u

            # Derivatives with respect to the spatial parameters
            for p0_shape_ind in 1:length(p0_shape)
                E_G_s.d[p0_shape[p0_shape_ind], 1] +=
                    sb.E_l_a[b, i].v[1] * a_i * fsm_i.d[p0_shape_ind, 1]
                E_G2_s.d[p0_shape[p0_shape_ind], 1] +=
                    sb.E_ll_a[b, i].v[1] * 2 * fsm_i.v[1] * a_i * fsm_i.d[p0_shape_ind, 1]
            end

            # Derivatives with respect to the brightness parameters.
            for p0_bright_ind in 1:length(p0_bright)
                E_G_s.d[p0_bright[p0_bright_ind], 1] +=
                    a_i * fsm_i.v[1] * sb.E_l_a[b, i].d[p0_bright_ind, 1]
                E_G2_s.d[p0_bright[p0_bright_ind], 1] +=
                    a_i * (fsm_i.v[1]^2) * sb.E_ll_a[b, i].d[p0_bright_ind, 1]
            end

            if elbo_vars.calculate_hessian
                ######################
                # Hessians.

                # Data structures to accumulate certain submatrices of the Hessian.
                E_G_s_hsub = elbo_vars.E_G_s_hsub_vec[i];
                E_G2_s_hsub = elbo_vars.E_G2_s_hsub_vec[i];

                # The (a, a) block of the hessian is zero.

                # The (bright, bright) block:
                for p0_ind1 in 1:length(p0_bright), p0_ind2 in 1:length(p0_bright)
                    # TODO: time consuming **************
                    E_G_s.h[p0_bright[p0_ind1], p0_bright[p0_ind2]] =
                        a_i * sb.E_l_a[b, i].h[p0_ind1, p0_ind2] * fsm_i.v[1]
                    E_G2_s.h[p0_bright[p0_ind1], p0_bright[p0_ind2]] =
                        (fsm_i.v[1]^2) * a_i * sb.E_ll_a[b, i].h[p0_ind1, p0_ind2]
                end

                # The (shape, shape) block:
                p1, p2 = size(E_G_s_hsub.shape_shape)
                for ind1 = 1:p1, ind2 = 1:p2
                    E_G_s_hsub.shape_shape[ind1, ind2] =
                        a_i * sb.E_l_a[b, i].v[1] * fsm_i.h[ind1, ind2]
                    E_G2_s_hsub.shape_shape[ind1, ind2] =
                        2 * a_i * sb.E_ll_a[b, i].v[1] * (
                            fsm_i.v[1] * fsm_i.h[ind1, ind2] +
                            fsm_i.d[ind1, 1] * fsm_i.d[ind2, 1])
                end

                # The u_u submatrix of this assignment will be overwritten after
                # the loop.
                for p0_ind1 in 1:length(p0_shape), p0_ind2 in 1:length(p0_shape)
                    E_G_s.h[p0_shape[p0_ind1], p0_shape[p0_ind2]] =
                        a_i * sb.E_l_a[b, i].v[1] * fsm_i.h[p0_ind1, p0_ind2]
                    E_G2_s.h[p0_shape[p0_ind1], p0_shape[p0_ind2]] =
                        E_G2_s_hsub.shape_shape[p0_ind1, p0_ind2];
                end

                # Since the u_u submatrix is not disjoint between different i, accumulate
                # it separate and add it at the end.
                for u_ind1 = 1:2, u_ind2 = 1:2
                    E_G_s_hsub.u_u[u_ind1, u_ind2] =
                        E_G_s_hsub.shape_shape[u_ind[u_ind1], u_ind[u_ind2]]
                    E_G2_s_hsub.u_u[u_ind1, u_ind2] =
                        E_G2_s_hsub.shape_shape[u_ind[u_ind1], u_ind[u_ind2]]
                end

                # All other terms are disjoint between different i and don't involve
                # addition, so we can just assign their values (which is efficient in
                # native julia).

                # The (a, bright) blocks:
                for p0_ind in 1:length(p0_bright)
                    E_G_s.h[p0_bright[p0_ind], ids.a[i]] =
                        fsm_i.v[1] * sb.E_l_a[b, i].d[p0_ind, 1]
                    E_G2_s.h[p0_bright[p0_ind], ids.a[i]] =
                        (fsm_i.v[1] ^ 2) * sb.E_ll_a[b, i].d[p0_ind, 1]
                    E_G_s.h[ids.a[i], p0_bright[p0_ind]] =
                        E_G_s.h[p0_bright[p0_ind], ids.a[i]]
                    E_G2_s.h[ids.a[i], p0_bright[p0_ind]] =
                        E_G2_s.h[p0_bright[p0_ind], ids.a[i]]
                end

                # The (a, shape) blocks.
                for p0_ind in 1:length(p0_shape)
                    E_G_s.h[p0_shape[p0_ind], ids.a[i]] =
                        sb.E_l_a[b, i].v[1] * fsm_i.d[p0_ind, 1]
                    E_G2_s.h[p0_shape[p0_ind], ids.a[i]] =
                        sb.E_ll_a[b, i].v[1] * 2 * fsm_i.v[1] * fsm_i.d[p0_ind, 1]
                    E_G_s.h[ids.a[i], p0_shape[p0_ind]] =
                        E_G_s.h[p0_shape[p0_ind], ids.a[i]]
                    E_G2_s.h[ids.a[i], p0_shape[p0_ind]] =
                        E_G2_s.h[p0_shape[p0_ind], ids.a[i]]
                end

                for ind_b in 1:length(p0_bright), ind_s in 1:length(p0_shape)
                    E_G_s.h[p0_bright[ind_b], p0_shape[ind_s]] =
                        a_i * sb.E_l_a[b, i].d[ind_b, 1] * fsm_i.d[ind_s, 1]
                    E_G2_s.h[p0_bright[ind_b], p0_shape[ind_s]] =
                        2 * a_i * sb.E_ll_a[b, i].d[ind_b, 1] * fsm_i.v[1] * fsm_i.d[ind_s]

                    E_G_s.h[p0_shape[ind_s], p0_bright[ind_b]] =
                        E_G_s.h[p0_bright[ind_b], p0_shape[ind_s]]
                    E_G2_s.h[p0_shape[ind_s], p0_bright[ind_b]] =
                        E_G2_s.h[p0_bright[ind_b], p0_shape[ind_s]]
                end
            end # if calculate hessian
        end # if calculate derivatives
    end # i loop

    if elbo_vars.calculate_hessian
        # Accumulate the u Hessian. u is the only parameter that is shared between
        # different values of i.

        # This is
        # for i = 1:Ia
        #     E_G_u_u_hess += elbo_vars.E_G_s_hsub_vec[i].u_u
        #     E_G2_u_u_hess += elbo_vars.E_G2_s_hsub_vec[i].u_u
        # end
        # For each value in 1:Ia, written this way for speed.
        @assert Ia == 2
        for u_ind1 = 1:2, u_ind2 = 1:2
            E_G_s.h[ids.u[u_ind1], ids.u[u_ind2]] =
            elbo_vars.E_G_s_hsub_vec[1].u_u[u_ind1, u_ind2] +
            elbo_vars.E_G_s_hsub_vec[2].u_u[u_ind1, u_ind2]

            E_G2_s.h[ids.u[u_ind1], ids.u[u_ind2]] =
                elbo_vars.E_G2_s_hsub_vec[1].u_u[u_ind1, u_ind2] +
                elbo_vars.E_G2_s_hsub_vec[2].u_u[u_ind1, u_ind2]
        end
    end

    calculate_var_G_s!(elbo_vars, active_source)

    true # Set default return type
end


"""
Calculate the variance var_G_s as a function of (E_G_s, E_G2_s).

Args:
    - elbo_vars: Elbo intermediate values.
    - active_source: Whether this is an active source that requires derivatives

Returns:
    Updates elbo_vars.var_G_s in place.
"""
function calculate_var_G_s!{NumType <: Number}(
                    elbo_vars::ElboIntermediateVariables{NumType},
                    active_source::Bool)
    var_G_s = elbo_vars.var_G_s
    E_G_s = elbo_vars.E_G_s
    E_G2_s = elbo_vars.E_G2_s

    clear!(var_G_s,
        elbo_vars.calculate_hessian &&
            elbo_vars.calculate_derivs && active_source)

    elbo_vars.var_G_s.v[1] = E_G2_s.v[1] - (E_G_s.v[1] ^ 2);

    if active_source && elbo_vars.calculate_derivs
        @assert length(var_G_s.d) == length(E_G2_s.d) == length(E_G_s.d)
        @inbounds for ind1 = 1:length(var_G_s.d)
            var_G_s.d[ind1] = E_G2_s.d[ind1] - 2 * E_G_s.v[1] * E_G_s.d[ind1]
        end

        if elbo_vars.calculate_hessian
            p1, p2 = size(var_G_s.h)
            @inbounds for ind2 = 1:p2, ind1 = 1:ind2
                var_G_s.h[ind1, ind2] =
                    E_G2_s.h[ind1, ind2] - 2 * (
                        E_G_s.v[1] * E_G_s.h[ind1, ind2] +
                        E_G_s.d[ind1, 1] * E_G_s.d[ind2, 1])
                var_G_s.h[ind2, ind1] = var_G_s.h[ind1, ind2]
            end
        end
    end
end


"""
Adds up E_G and var_G across all sources.

Args:
    - elbo_vars: Elbo intermediate values, with updated fs1m_vec and fs0m_vec.
    - mp: Model parameters
    - tile_sources: A vector of integers of sources in 1:mp.S affecting the tile
    - tile: An ImageTile
    - sbs: Source brightnesses

Updates elbo_vars.E_G and elbo_vars.var_G in place.
"""
function combine_pixel_sources!{NumType <: Number}(
                    elbo_vars::ElboIntermediateVariables{NumType},
                    mp::ModelParams{NumType},
                    tile_sources::Vector{Int},
                    tile::ImageTile,
                    sbs::Vector{SourceBrightness{NumType}})
    clear!(elbo_vars.E_G,
        elbo_vars.calculate_hessian && elbo_vars.calculate_derivs)
    clear!(elbo_vars.var_G,
        elbo_vars.calculate_hessian && elbo_vars.calculate_derivs)

    for s in tile_sources
        active_source = s in mp.active_sources
        calculate_hessian =
            elbo_vars.calculate_hessian && elbo_vars.calculate_derivs &&
            active_source
        accumulate_source_brightness!(elbo_vars, mp, sbs, s, tile.b)
        if active_source
            sa = findfirst(mp.active_sources, s)
            add_sources_sf!(elbo_vars.E_G, elbo_vars.E_G_s, sa, calculate_hessian)
            add_sources_sf!(elbo_vars.var_G, elbo_vars.var_G_s, sa, calculate_hessian)
        else
            # If the sources is inactives, simply accumulate the values.
            elbo_vars.E_G.v[1] += elbo_vars.E_G_s.v[1]
            elbo_vars.var_G.v[1] += elbo_vars.var_G_s.v[1]
        end
    end
end


"""
Expected brightness for a single pixel.

Args:
    - elbo_vars: Elbo intermediate values.
    - h, w: The integer locations of the pixel within the tile
    - sbs: Source brightnesses
    - star_mcs: Star components
    - gal_mcs: Galaxy components
    - tile: An ImageTile
    - mp: Model parameters
    - tile_sources: A vector of integers of sources in 1:mp.S affecting the tile
    - include_epsilon: Whether the background noise should be included

Returns:
    - Updates elbo_vars.E_G and elbo_vars.var_G in place.
"""
function get_expected_pixel_brightness!{NumType <: Number}(
                    elbo_vars::ElboIntermediateVariables{NumType},
                    h::Int, w::Int,
                    sbs::Vector{SourceBrightness{NumType}},
                    star_mcs::Array{BvnComponent{NumType}, 2},
                    gal_mcs::Array{GalaxyCacheComponent{NumType}, 4},
                    tile::ImageTile,
                    mp::ModelParams{NumType},
                    tile_sources::Vector{Int};
                    include_epsilon::Bool=true)
    # This combines the bvn components to get the brightness for each
    # source separately.
    populate_fsm_vecs!(
        elbo_vars, mp, tile_sources, tile, h, w, sbs, gal_mcs, star_mcs)

    # # This combines the sources into a single brightness value for the pixel.
    combine_pixel_sources!(elbo_vars, mp, tile_sources, tile, sbs);

    if include_epsilon
        # There are no derivatives with respect to epsilon, so can safely add
        # to the value.
        elbo_vars.E_G.v[1] += tile.epsilon_mat[h, w]
    end
end


"""
Add the lower bound to the log term to the elbo for a single pixel.

Args:
     - elbo_vars: Intermediate variables
     - x_nbm: The photon count at this pixel
     - iota: The optical sensitivity

 Returns:
    Updates elbo_vars.elbo in place by adding the lower bound to the log
    term.
"""
function add_elbo_log_term!{NumType <: Number}(
        elbo_vars::ElboIntermediateVariables{NumType},
        x_nbm::AbstractFloat, iota::AbstractFloat)

    # See notes for a derivation. The log term is
    # log E[G] - Var(G) / (2 * E[G] ^2 )

    E_G = elbo_vars.E_G
    var_G = elbo_vars.var_G
    elbo = elbo_vars.elbo

    # The gradients and Hessians are written as a f(x, y) = f(E_G2, E_G)
    log_term_value = log(E_G.v[1]) - 0.5 * var_G.v[1]    / (E_G.v[1] ^ 2)

    # Add x_nbm * (log term * log(iota)) to the elbo.
    # If not calculating derivatives, add the values directly.
    elbo.v[1] += x_nbm * (log(iota) + log_term_value)

    if elbo_vars.calculate_derivs
        elbo_vars.combine_grad[1] = -0.5 / (E_G.v[1] ^ 2)
        elbo_vars.combine_grad[2] = 1 / E_G.v[1] + var_G.v[1] / (E_G.v[1] ^ 3)

        if elbo_vars.calculate_hessian
            elbo_vars.combine_hess[1, 1] = 0.0
            elbo_vars.combine_hess[1, 2] = elbo_vars.combine_hess[2, 1] = 1 / E_G.v[1]^3
            elbo_vars.combine_hess[2, 2] =
                -(1 / E_G.v[1] ^ 2 + 3    * var_G.v[1] / (E_G.v[1] ^ 4))
        end

        # Calculate the log term.
        combine_sfs!(
            elbo_vars.var_G, elbo_vars.E_G, elbo_vars.elbo_log_term,
            log_term_value, elbo_vars.combine_grad, elbo_vars.combine_hess,
            calculate_hessian=elbo_vars.calculate_hessian)

        # Add to the ELBO.
        for ind in 1:length(elbo.d)
            elbo.d[ind] += x_nbm * elbo_vars.elbo_log_term.d[ind]
        end

        if elbo_vars.calculate_hessian
            for ind in 1:length(elbo.h)
                elbo.h[ind] += x_nbm * elbo_vars.elbo_log_term.h[ind]
            end
        end
    end
end


############################################
# The remaining functions loop over tiles, sources, and pixels.

"""
Add an array of pixels' contribution to the ELBO likelihood term by
modifying elbo in place.

Args:
    - elbo_vars: Elbo intermediate values.
    - tiled_blob: An Array of images
    - mp: Model parameters
    - active_pixels: An array of ActivePixels to be processed.

Returns:
    Adds to elbo_vars.elbo in place.
"""
function process_active_pixels!{NumType <: Number}(
        elbo_vars::ElboIntermediateVariables{NumType},
        tiled_blob::TiledBlob,
        mp::ModelParams{NumType},
        active_pixels::Array{ActivePixel})

    sbs = load_source_brightnesses(mp,
        calculate_derivs=elbo_vars.calculate_derivs,
        calculate_hessian=elbo_vars.calculate_hessian)

    star_mcs_vec = Array(Array{BvnComponent{NumType}, 2}, length(tiled_blob))
    gal_mcs_vec = Array(Array{GalaxyCacheComponent{NumType}, 4}, length(tiled_blob))

    for b=1:length(tiled_blob)
        # TODO: every elbo_vars should not get to decide its own calculate_*
        star_mcs_vec[b], gal_mcs_vec[b] =
            load_bvn_mixtures(mp, b,
                calculate_derivs=elbo_vars.calculate_derivs,
                calculate_hessian=elbo_vars.calculate_hessian)
    end

    # iterate over the pixels
    for pixel in active_pixels
        tile = tiled_blob[pixel.b].tiles[pixel.tile_ind]
        tile_sources = mp.tile_sources[pixel.b][pixel.tile_ind]
        this_pixel = tile.pixels[pixel.h, pixel.w]

        # Get the brightness.
        get_expected_pixel_brightness!(
            elbo_vars, pixel.h, pixel.w, sbs,
            star_mcs_vec[pixel.b], gal_mcs_vec[pixel.b], tile,
            mp, tile_sources, include_epsilon=true)

        # Add the terms to the elbo given the brightness.
        iota = tile.iota_vec[pixel.h]
        add_elbo_log_term!(elbo_vars, this_pixel, iota)
        add_scaled_sfs!(elbo_vars.elbo,
                                        elbo_vars.E_G, -iota,
                                        elbo_vars.calculate_hessian &&
                                        elbo_vars.calculate_derivs)

        # Subtract the log factorial term. This is not a function of the
        # parameters so the derivatives don't need to be updated. Note that
        # even though this does not affect the ELBO's maximum, it affects
        # the optimization convergence criterion, so I will leave it in for now.
        elbo_vars.elbo.v[1] -= lfact(this_pixel)
    end
end


"""
Return the image predicted for the tile given the current parameters.

Args:
    - elbo_vars: Elbo intermediate values.
    - tile: An ImageTile
    - mp: Model parameters
    - tile_sources: A vector of integers of sources in 1:mp.S affecting the tile
    - sbs: Source brightnesses
    - star_mcs: Star components
    - gal_mcs: Galaxy components
    - include_epsilon: Whether the background noise should be included

Returns:
    A matrix of the same size as the tile with the predicted brightnesses.
"""
function tile_predicted_image{NumType <: Number}(
                elbo_vars::ElboIntermediateVariables{NumType},
                tile::ImageTile,
                mp::ModelParams{NumType},
                tile_sources::Vector{Int},
                sbs::Vector{SourceBrightness{NumType}},
                star_mcs::Array{BvnComponent{NumType}, 2},
                gal_mcs::Array{GalaxyCacheComponent{NumType}, 4};
                include_epsilon::Bool=true)

        predicted_pixels = copy(tile.pixels)
        # Iterate over pixels that are not NaN.
        for w in 1:tile.w_width, h in 1:tile.h_width
            this_pixel = tile.pixels[h, w]
            if !Base.isnan(this_pixel)
                get_expected_pixel_brightness!(
                    elbo_vars, h, w, sbs, star_mcs, gal_mcs, tile,
                    mp, tile_sources, include_epsilon=include_epsilon)
                iota = tile.iota_vec[h]
                predicted_pixels[h, w] = elbo_vars.E_G.v[1] * iota
            end
        end

        predicted_pixels
end


"""
Produce a predicted image for a given tile and model parameters.

Args:
    - tile: An ImageTile
    - mp: Model parameters
    - tile_sources: A vector of integers of sources in 1:mp.S affecting the tile
    - include_epsilon: Whether the background noise should be included

If include_epsilon is true, then the background is also rendered.
Otherwise, only pixels from the object are rendered.
"""
function tile_predicted_image{NumType <: Number}(
                    tile::ImageTile, mp::ModelParams{NumType},
                    tile_sources::Vector{Int};
                    include_epsilon::Bool=false)
    star_mcs, gal_mcs = load_bvn_mixtures(mp, tile.b, calculate_derivs=false);
    sbs = load_source_brightnesses(mp, calculate_derivs=false);

    elbo_vars =
        ElboIntermediateVariables(NumType, mp.S, length(mp.active_sources));
    elbo_vars.calculate_derivs = false

    tile_predicted_image(
        elbo_vars, tile, mp, tile_sources, sbs, star_mcs, gal_mcs,
        include_epsilon=include_epsilon)
end


"""
Get the active pixels (pixels for which the active sources are present)
for a tiled blob.

TODO: move this to pre-processing and use it instead of setting low-signal
pixels to NaN.
"""
function get_active_pixels{NumType <: Number}(
                    tiled_blob::TiledBlob,
                    mp::ModelParams{NumType})
    active_pixels = ActivePixel[]
    for b in 1:length(tiled_blob), tile_ind in 1:length(tiled_blob[b].tiles)
        tile_sources = mp.tile_sources[b][tile_ind]
        if length(intersect(tile_sources, mp.active_sources)) > 0
            tile = tiled_blob[b].tiles[tile_ind]
            for w in 1:tile.w_width, h in 1:tile.h_width
                if !Base.isnan(tile.pixels[h, w])
                    push!(active_pixels, ActivePixel(b, tile_ind, h, w))
                end
            end
        end
    end

    active_pixels
end


"""
Return the expected log likelihood for all bands in a section
of the sky.

Args:
    - tiled_blob: A TiledBlob
    - mp: Model parameters
    - calculate_derivs: Whether or not any gradient or hessian information will
                        be calculated
    - calculate_hessian: Whether to calculate a Hessian. If calculate_derivs is
                         false, a Hessian will not be calculated irrespective of
                         the value of calculate_hessian.

Returns:
    A sensitive float with the log likelihood.
"""
function elbo_likelihood{NumType <: Number}(
                    tiled_blob::TiledBlob,
                    mp::ModelParams{NumType};
                    calculate_derivs::Bool=true,
                    calculate_hessian::Bool=true)
    active_pixels = get_active_pixels(tiled_blob, mp)
    elbo_vars = ElboIntermediateVariables(NumType, mp.S,
                                        length(mp.active_sources), calculate_derivs=calculate_derivs,
                                        calculate_hessian=calculate_hessian)
    process_active_pixels!(elbo_vars, tiled_blob, mp, active_pixels)
    elbo_vars.elbo
end


"""
Calculates and returns the ELBO and its derivatives for all the bands
of an image.

Args:
    - tiled_blob: A TiledBlob.
    - mp: Model parameters.
    - calculate_derivs: Whether or not any gradient or hessian information will
                        be calculated
    - calculate_hessian: Whether to calculate a Hessian. If calculate_derivs is
                         false, a Hessian will not be calculated irrespective of
                         the value of calculate_hessian.

Returns:
    A sensitive float containing the ELBO for the image.
"""
function elbo{NumType <: Number}(
                    tiled_blob::TiledBlob,
                    mp::ModelParams{NumType};
                    calculate_derivs::Bool=true,
                    calculate_hessian::Bool=true)
    elbo = elbo_likelihood(tiled_blob, mp,
        calculate_derivs=calculate_derivs, calculate_hessian=calculate_hessian)

    # TODO: subtract the kl with the hessian.
    subtract_kl!(mp, elbo,
        calculate_derivs=calculate_derivs, calculate_hessian=calculate_hessian)
    elbo
end


end
