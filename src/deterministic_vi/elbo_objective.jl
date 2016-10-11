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
                    ea::ElboArgs{NumType},
                    b::Int;
                    calculate_derivs::Bool=true,
                    calculate_hessian::Bool=true)
    # call bvn loader from the Model Module
    Model.load_bvn_mixtures(ea.S, ea.patches, ea.vp, ea.active_sources,
                            ea.psf_K, b,
                            calculate_derivs=calculate_derivs,
                            calculate_hessian=calculate_hessian)
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
    - is_active_source: Whether it is an active source, (i.e. whether to
                        calculate derivatives if requested.)

Returns:
    Updates elbo_vars.fs0m_vec[s] in place.
"""
function accum_star_pos!{NumType <: Number}(
                    elbo_vars::ElboIntermediateVariables{NumType},
                    s::Int,
                    bmc::BvnComponent{NumType},
                    x::Vector{Float64},
                    wcs_jacobian::Array{Float64, 2},
                    is_active_source::Bool)
    # call accum star pos in model
    Model.accum_star_pos!(elbo_vars.bvn_derivs,
                    elbo_vars.fs0m_vec,
                    elbo_vars.calculate_derivs,
                    elbo_vars.calculate_hessian,
                    s, bmc, x, wcs_jacobian, is_active_source)
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
    - is_active_source: Whether it is an active source, (i.e. whether to
                        calculate derivatives if requested.)

Returns:
    Updates elbo_vars.fs1m_vec[s] in place.
"""
function accum_galaxy_pos!{NumType <: Number}(
                    elbo_vars::ElboIntermediateVariables{NumType},
                    s::Int,
                    gcc::GalaxyCacheComponent{NumType},
                    x::Vector{Float64},
                    wcs_jacobian::Array{Float64, 2},
                    is_active_source::Bool)
    # call accum star pos in model
    Model.accum_galaxy_pos!(elbo_vars.bvn_derivs,
                            elbo_vars.fs1m_vec,
                            elbo_vars.calculate_derivs,
                            elbo_vars.calculate_hessian,
                            s, gcc, x, wcs_jacobian, is_active_source)
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
    Model.populate_fsm_vecs!(elbo_vars.bvn_derivs,
                             elbo_vars.fs0m_vec,
                             elbo_vars.fs1m_vec,
                             elbo_vars.calculate_derivs,
                             elbo_vars.calculate_hessian,
                             ea.patches, ea.active_sources,
                             ea.psf_K, ea.num_allowed_sd,
                             tile_sources, tile, h, w, gal_mcs, star_mcs)
end


"""
Add the contributions of a single source to E_G_s and var_G_s, which are cleared
and then updated in place.

Args:
    - elbo_vars: Elbo intermediate values, with updated fs1m_vec and fs0m_vec.
    - ea: Model parameters
    - sbs: Source brightnesses
    - s: The source, in 1:ea.S
    - b: The band

Returns:
    Updates elbo_vars.E_G_s and elbo_vars.var_G_s in place with the brightness
    for this sourve at this pixel.
"""
function accumulate_source_brightness!{NumType <: Number}(
                    elbo_vars::ElboIntermediateVariables{NumType},
                    ea::ElboArgs{NumType},
                    sbs::Vector{SourceBrightness{NumType}},
                    s::Int, b::Int)
    # E[G] and E{G ^ 2} for a single source
    E_G_s = elbo_vars.E_G_s
    E_G2_s = elbo_vars.E_G2_s

    clear_hessian = elbo_vars.calculate_hessian && elbo_vars.calculate_derivs
    clear!(E_G_s, clear_hessian)
    clear!(E_G2_s, clear_hessian)

    sb = sbs[s]

    active_source = (s in ea.active_sources)

    for i in 1:Ia # Stars and galaxies
        fsm_i = (i == 1) ? elbo_vars.fs0m_vec[s] : elbo_vars.fs1m_vec[s]
        a_i = ea.vp[s][ids.a[i, 1]]

        lf = sb.E_l_a[b, i].v[1] * fsm_i.v[1]
        llff = sb.E_ll_a[b, i].v[1] * fsm_i.v[1]^2

        E_G_s.v[1] += a_i * lf
        E_G2_s.v[1] += a_i * llff

        # Only calculate derivatives for active sources.
        if active_source && elbo_vars.calculate_derivs
            ######################
            # Gradients.

            E_G_s.d[ids.a[i, 1], 1] += lf
            E_G2_s.d[ids.a[i, 1], 1] += llff

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
                E_G_s_hsub = elbo_vars.E_G_s_hsub_vec[i]
                E_G2_s_hsub = elbo_vars.E_G2_s_hsub_vec[i]

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
                        E_G2_s_hsub.shape_shape[p0_ind1, p0_ind2]
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
                    E_G_s.h[p0_bright[p0_ind], ids.a[i, 1]] =
                        fsm_i.v[1] * sb.E_l_a[b, i].d[p0_ind, 1]
                    E_G2_s.h[p0_bright[p0_ind], ids.a[i, 1]] =
                        (fsm_i.v[1] ^ 2) * sb.E_ll_a[b, i].d[p0_ind, 1]
                    E_G_s.h[ids.a[i, 1], p0_bright[p0_ind]] =
                        E_G_s.h[p0_bright[p0_ind], ids.a[i, 1]]
                    E_G2_s.h[ids.a[i, 1], p0_bright[p0_ind]] =
                        E_G2_s.h[p0_bright[p0_ind], ids.a[i, 1]]
                end

                # The (a, shape) blocks.
                for p0_ind in 1:length(p0_shape)
                    E_G_s.h[p0_shape[p0_ind], ids.a[i, 1]] =
                        sb.E_l_a[b, i].v[1] * fsm_i.d[p0_ind, 1]
                    E_G2_s.h[p0_shape[p0_ind], ids.a[i, 1]] =
                        sb.E_ll_a[b, i].v[1] * 2 * fsm_i.v[1] * fsm_i.d[p0_ind, 1]
                    E_G_s.h[ids.a[i, 1], p0_shape[p0_ind]] =
                        E_G_s.h[p0_shape[p0_ind], ids.a[i, 1]]
                    E_G2_s.h[ids.a[i, 1], p0_shape[p0_ind]] =
                        E_G2_s.h[p0_shape[p0_ind], ids.a[i, 1]]
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

    elbo_vars.var_G_s.v[1] = E_G2_s.v[1] - (E_G_s.v[1] ^ 2)

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
    - ea: Model parameters
    - tile_sources: A vector of integers of sources in 1:ea.S affecting the tile
    - tile: An ImageTile
    - sbs: Source brightnesses

Updates elbo_vars.E_G and elbo_vars.var_G in place.
"""
function combine_pixel_sources!{NumType <: Number}(
                    elbo_vars::ElboIntermediateVariables{NumType},
                    ea::ElboArgs{NumType},
                    tile_sources::Vector{Int},
                    tile::ImageTile,
                    sbs::Vector{SourceBrightness{NumType}})
    clear!(elbo_vars.E_G,
        elbo_vars.calculate_hessian && elbo_vars.calculate_derivs)
    clear!(elbo_vars.var_G,
        elbo_vars.calculate_hessian && elbo_vars.calculate_derivs)

    for s in tile_sources
        active_source = s in ea.active_sources
        calculate_hessian =
            elbo_vars.calculate_hessian && elbo_vars.calculate_derivs &&
            active_source
        accumulate_source_brightness!(elbo_vars, ea, sbs, s, tile.b)
        if active_source
            sa = findfirst(ea.active_sources, s)
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
    - ea: Model parameters
    - tile_sources: A vector of integers of sources in 1:ea.S affecting the tile
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
                ea::ElboArgs{NumType},
                tile_sources::Vector{Int};
                include_epsilon::Bool=true)
    # This combines the bvn components to get the brightness for each
    # source separately.
    populate_fsm_vecs!(
        elbo_vars, ea, tile_sources, tile, h, w, gal_mcs, star_mcs)

    # # This combines the sources into a single brightness value for the pixel.
    combine_pixel_sources!(elbo_vars, ea, tile_sources, tile, sbs)

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
    - ea: Model parameters

Returns:
    Adds to elbo_vars.elbo in place.
"""
function process_active_pixels!{NumType <: Number}(
                elbo_vars::ElboIntermediateVariables{NumType},
                ea::ElboArgs{NumType})
    sbs = load_source_brightnesses(ea,
        calculate_derivs=elbo_vars.calculate_derivs,
        calculate_hessian=elbo_vars.calculate_hessian)

    star_mcs_vec = Array(Array{BvnComponent{NumType}, 2}, ea.N)
    gal_mcs_vec = Array(Array{GalaxyCacheComponent{NumType}, 4}, ea.N)

    for b=1:ea.N
        star_mcs_vec[b], gal_mcs_vec[b] =
            load_bvn_mixtures(ea, b,
                calculate_derivs=elbo_vars.calculate_derivs,
                calculate_hessian=elbo_vars.calculate_hessian)
    end

    # iterate over the pixels
    for pixel in ea.active_pixels
        tile = ea.images[pixel.n].tiles[pixel.tile_ind]
        tile_sources = ea.tile_source_map[pixel.n][pixel.tile_ind]
        this_pixel = tile.pixels[pixel.h, pixel.w]

        # Get the brightness.
        get_expected_pixel_brightness!(
            elbo_vars, pixel.h, pixel.w, sbs,
            star_mcs_vec[pixel.n], gal_mcs_vec[pixel.n], tile,
            ea, tile_sources, include_epsilon=true)

        # Add the terms to the elbo given the brightness.
        iota = tile.iota_vec[pixel.h]
        add_elbo_log_term!(elbo_vars, this_pixel, iota)
        add_scaled_sfs!(elbo_vars.elbo,
                        elbo_vars.E_G,
                        -iota,
                        elbo_vars.calculate_hessian && elbo_vars.calculate_derivs)

        # Subtract the log factorial term. This is not a function of the
        # parameters so the derivatives don't need to be updated. Note that
        # even though this does not affect the ELBO's maximum, it affects
        # the optimization convergence criterion, so I will leave it in for now.
        elbo_vars.elbo.v[1] -= lfact(this_pixel)
    end
    assert_all_finite(elbo_vars.elbo)
end


"""
Return the image predicted for the tile given the current parameters.

Args:
    - elbo_vars: Elbo intermediate values.
    - tile: An ImageTile
    - ea: Model parameters
    - tile_sources: A vector of integers of sources in 1:ea.S affecting the tile
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
                ea::ElboArgs{NumType},
                tile_sources::Vector{Int},
                sbs::Vector{SourceBrightness{NumType}},
                star_mcs::Array{BvnComponent{NumType}, 2},
                gal_mcs::Array{GalaxyCacheComponent{NumType}, 4};
                include_epsilon::Bool=true)
    predicted_pixels = copy(tile.pixels)
    # Iterate over pixels that are not NaN.
    h_width, w_width = size(tile.pixels)
    for w in 1:w_width, h in 1:h_width
        this_pixel = tile.pixels[h, w]
        if !Base.isnan(this_pixel)
            get_expected_pixel_brightness!(
                elbo_vars, h, w, sbs, star_mcs, gal_mcs, tile,
                ea, tile_sources, include_epsilon=include_epsilon)
            iota = tile.iota_vec[h]
            predicted_pixels[h, w] = elbo_vars.E_G.v[1] * iota
        end
    end

    predicted_pixels
end


"""
Produce a predicted image for a given tile and model parameters.
If include_epsilon is true, then the background is also rendered.
Otherwise, only pixels from the object are rendered.

Args:
    - tile: An ImageTile
    - ea: Model parameters
    - tile_sources: A vector of integers of sources in 1:ea.S affecting the tile
    - include_epsilon: Whether the background noise should be included
"""
function tile_predicted_image{NumType <: Number}(
                    tile::ImageTile,
                    ea::ElboArgs{NumType},
                    tile_sources::Vector{Int};
                    include_epsilon::Bool=false)
    star_mcs, gal_mcs = load_bvn_mixtures(ea, tile.b, calculate_derivs=false)
    sbs = load_source_brightnesses(ea, calculate_derivs=false)

    clear!(ea.elbo_vars)
    ea.elbo_vars.calculate_derivs = false
    ea.elbo_vars.calculate_hessian = false

    tile_predicted_image(
        ea.elbo_vars, tile, ea, tile_sources, sbs, star_mcs, gal_mcs,
        include_epsilon=include_epsilon)
end


"""
Return the expected log likelihood for all bands in a section
of the sky.
Returns: A sensitive float with the log,  likelihood.
"""
function elbo_likelihood{NumType <: Number}(
                    ea::ElboArgs{NumType};
                    calculate_derivs=true,
                    calculate_hessian=true)
    clear!(ea.elbo_vars)
    ea.elbo_vars.calculate_derivs = calculate_derivs
    ea.elbo_vars.calculate_hessian = calculate_derivs && calculate_hessian

    process_active_pixels!(ea.elbo_vars, ea)
    deepcopy(ea.elbo_vars.elbo)
end


"""
Calculates and returns the ELBO and its derivatives for all the bands
of an image.
Returns: A sensitive float containing the ELBO for the image.
"""
function elbo{NumType <: Number}(
                    ea::ElboArgs{NumType};
                    calculate_derivs=true,
                    calculate_hessian=true)
    elbo = elbo_likelihood(ea; calculate_derivs=calculate_derivs,
                               calculate_hessian=calculate_hessian)
    # TODO: subtract the kl with the hessian.
    subtract_kl!(ea, elbo, calculate_derivs=calculate_derivs)
    elbo
end

