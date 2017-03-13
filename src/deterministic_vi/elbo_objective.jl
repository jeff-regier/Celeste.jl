"""
Calculate the contributions of a single source for a single pixel to
the sensitive floats E_G_s and var_G_s, which are cleared and updated in place.

Args:
    - elbo_vars: Elbo intermediate values, with updated fs1m and fs0m.
    - ea: Model parameters
    - E_G_s, var_G_s: The expectatio and variance of the brightnesses of this
          source at this pixel, updated in place.
    - fs0m, fs1m: The star and galaxy shape parameters for this source at
          this pixel.
    - sb: Source brightnesse
    - s: The source, in 1:ea.S
    - b: The band

Returns:
    Updates E_G_s and var_G_s in place with the brightness
    for this source at this pixel.
"""
function calculate_E_G_s!{NumType <: Number}(
                    ea::ElboArgs,
                    vp::VariationalParams{NumType},
                    elbo_vars::ElboIntermediateVariables{NumType},
                    sb::SourceBrightness{NumType},
                    b::Int,
                    s::Int,
                    is_active_source::Bool)
    E_G_s = elbo_vars.E_G_s
    E_G2_s = elbo_vars.E_G2_s
    var_G_s = elbo_vars.var_G_s
    fs0m = elbo_vars.fs0m
    fs1m = elbo_vars.fs1m

    @assert E_G_s.local_P == var_G_s.local_P == length(CanonicalParams)
    @assert E_G_s.local_S == var_G_s.local_S == 1
    @assert fs0m.local_P == length(StarPosParams)
    @assert fs1m.local_P == length(GalaxyPosParams)

    clear!(E_G_s)
    clear!(E_G2_s)

    @inbounds for i in 1:Ia # Celestial object types (e.g. stars and galaxies)
        fsm_i = (i == 1) ? fs0m : fs1m
        a_i = vp[s][ids.a[i]]
        sb_E_l_a_b_i = sb.E_l_a[b, i]
        sb_E_ll_a_b_i = sb.E_ll_a[b, i]

        fsm_i_v = fsm_i.v[]
        sb_E_l_a_b_i_v = sb_E_l_a_b_i.v[]
        sb_E_ll_a_b_i_v = sb_E_ll_a_b_i.v[]

        fsm_i_d = fsm_i.d
        sb_E_l_a_b_i_d = sb_E_l_a_b_i.d
        sb_E_ll_a_b_i_d = sb_E_ll_a_b_i.d
        E_G_s_d = E_G_s.d
        E_G2_s_d = E_G2_s.d

        E_G_s_h = E_G_s.h
        E_G2_s_h = E_G2_s.h

        lf = sb_E_l_a_b_i_v * fsm_i_v
        llff = sb_E_ll_a_b_i_v * fsm_i_v^2

        E_G_s.v[] += a_i * lf
        E_G2_s.v[] += a_i * llff

        ############ Only gradient and hessian code below ##############
        (is_active_source && elbo_vars.elbo.has_gradient) || continue

        E_G_s_d[ids.a[i], 1] += lf
        E_G2_s_d[ids.a[i], 1] += llff

        p0_shape = shape_standard_alignment[i]
        p0_bright = brightness_standard_alignment[i]
        u_ind = i == 1 ? star_ids.u : gal_ids.u

        # Derivatives with respect to the spatial parameters
        tmp1 = sb_E_l_a_b_i_v * a_i
        tmp2 = sb_E_ll_a_b_i_v * 2 * fsm_i_v * a_i
        for p0_shape_ind in 1:length(p0_shape)
            E_G_s_d[p0_shape[p0_shape_ind], 1] += tmp1 * fsm_i_d[p0_shape_ind, 1]
            E_G2_s_d[p0_shape[p0_shape_ind], 1] += tmp2 * fsm_i_d[p0_shape_ind, 1]
        end

        # Derivatives with respect to the brightness parameters.
        for p0_bright_ind in 1:length(p0_bright)
            E_G_s_d[p0_bright[p0_bright_ind], 1] +=
                a_i * fsm_i_v * sb_E_l_a_b_i_d[p0_bright_ind, 1]
            E_G2_s_d[p0_bright[p0_bright_ind], 1] +=
                a_i * (fsm_i_v^2) * sb_E_ll_a_b_i_d[p0_bright_ind, 1]
        end

        ############ only hessian code below ##############
        elbo_vars.elbo.has_hessian || continue

        E_G_s_hsub = elbo_vars.E_G_s_hsub_vec[i]
        E_G2_s_hsub = elbo_vars.E_G2_s_hsub_vec[i]

        # The (a, a) block of the hessian is zero.

        # The (bright, bright) block:
        for p0_ind1 in 1:length(p0_bright), p0_ind2 in 1:length(p0_bright)
            # TODO: time consuming **************
            E_G_s_h[p0_bright[p0_ind1], p0_bright[p0_ind2]] =
                a_i * sb_E_l_a_b_i.h[p0_ind1, p0_ind2] * fsm_i_v
            E_G2_s_h[p0_bright[p0_ind1], p0_bright[p0_ind2]] =
                (fsm_i_v^2) * a_i * sb_E_ll_a_b_i.h[p0_ind1, p0_ind2]
        end

        # The (shape, shape) block:
        p1, p2 = size(E_G_s_hsub.shape_shape)
        for ind1 = 1:p1, ind2 = 1:p2
            E_G_s_hsub.shape_shape[ind1, ind2] =
                a_i * sb_E_l_a_b_i_v * fsm_i.h[ind1, ind2]
            E_G2_s_hsub.shape_shape[ind1, ind2] =
                2 * a_i * sb_E_ll_a_b_i_v * (
                    fsm_i_v * fsm_i.h[ind1, ind2] +
                    fsm_i_d[ind1, 1] * fsm_i_d[ind2, 1])
        end

        # The u_u submatrix of this assignment will be overwritten after
        # the loop.
        for p0_ind1 in 1:length(p0_shape), p0_ind2 in 1:length(p0_shape)
            E_G_s_h[p0_shape[p0_ind1], p0_shape[p0_ind2]] =
                a_i * sb_E_l_a_b_i_v * fsm_i.h[p0_ind1, p0_ind2]
            E_G2_s_h[p0_shape[p0_ind1], p0_shape[p0_ind2]] =
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
            E_G_s_h[p0_bright[p0_ind], ids.a[i]] =
                fsm_i_v * sb_E_l_a_b_i_d[p0_ind, 1]
            E_G2_s_h[p0_bright[p0_ind], ids.a[i]] =
                (fsm_i_v ^ 2) * sb_E_ll_a_b_i_d[p0_ind, 1]
            E_G_s_h[ids.a[i], p0_bright[p0_ind]] =
                E_G_s_h[p0_bright[p0_ind], ids.a[i]]
            E_G2_s_h[ids.a[i], p0_bright[p0_ind]] =
                E_G2_s_h[p0_bright[p0_ind], ids.a[i]]
        end

        # The (a, shape) blocks.
        for p0_ind in 1:length(p0_shape)
            E_G_s_h[p0_shape[p0_ind], ids.a[i]] =
                sb_E_l_a_b_i_v * fsm_i_d[p0_ind, 1]
            E_G2_s_h[p0_shape[p0_ind], ids.a[i]] =
                sb_E_ll_a_b_i_v * 2 * fsm_i_v * fsm_i_d[p0_ind, 1]
            E_G_s_h[ids.a[i], p0_shape[p0_ind]] =
                E_G_s_h[p0_shape[p0_ind], ids.a[i]]
            E_G2_s_h[ids.a[i], p0_shape[p0_ind]] =
                E_G2_s_h[p0_shape[p0_ind], ids.a[i]]
        end

        for ind_b in 1:length(p0_bright), ind_s in 1:length(p0_shape)
            E_G_s_h[p0_bright[ind_b], p0_shape[ind_s]] =
                a_i * sb_E_l_a_b_i_d[ind_b, 1] * fsm_i_d[ind_s, 1]
            E_G2_s_h[p0_bright[ind_b], p0_shape[ind_s]] =
                2 * a_i * sb_E_ll_a_b_i_d[ind_b, 1] * fsm_i_v * fsm_i_d[ind_s]

            E_G_s_h[p0_shape[ind_s], p0_bright[ind_b]] =
                E_G_s_h[p0_bright[ind_b], p0_shape[ind_s]]
            E_G2_s_h[p0_shape[ind_s], p0_bright[ind_b]] =
                E_G2_s_h[p0_bright[ind_b], p0_shape[ind_s]]
        end
    end # i loop

    if elbo_vars.elbo.has_hessian
        # Accumulate the u Hessian. u is the only parameter that is shared between
        # different values of i.

        # This is
        # for i = 1:Ia
        #     E_G_u_u_hess += elbo_vars.E_G_s_hsub_vec[i].u_u
        #     E_G2_u_u_hess += elbo_vars.E_G2_s_hsub_vec[i].u_u
        # end
        # For each value in 1:Ia, written this way for speed.

        for u_ind1 = 1:2, u_ind2 = 1:2
            elbo_vars.E_G_s.h[ids.u[u_ind1], ids.u[u_ind2]] =
                elbo_vars.E_G_s_hsub_vec[1].u_u[u_ind1, u_ind2] +
                elbo_vars.E_G_s_hsub_vec[2].u_u[u_ind1, u_ind2]

            elbo_vars.E_G2_s.h[ids.u[u_ind1], ids.u[u_ind2]] =
                elbo_vars.E_G2_s_hsub_vec[1].u_u[u_ind1, u_ind2] +
                elbo_vars.E_G2_s_hsub_vec[2].u_u[u_ind1, u_ind2]
        end
    end
end


"""
Calculate the variance var_G_s as a function of (E_G_s, E_G2_s).

Args:
    - elbo_vars: Elbo intermediate values.
    - E_G_s: The expected brightness for a source
    - E_G2_s: The expected squared brightness for a source
    - var_G_s: Updated in place.  The variance of the brightness of a source.
    - is_active_source: Whether this is an active source that requires derivatives

Returns:
    Updates var_G_s in place.
"""
function calculate_var_G_s!{NumType <: Number}(
                    ea::ElboArgs,
                    vp::VariationalParams{NumType},
                    elbo_vars::ElboIntermediateVariables{NumType},
                    is_active_source::Bool)
    E_G_s = elbo_vars.E_G_s
    E_G2_s = elbo_vars.E_G2_s
    var_G_s = elbo_vars.var_G_s

    clear!(var_G_s)

    E_G_s_v = elbo_vars.E_G_s.v[]
    E_G2_s_v = elbo_vars.E_G2_s.v[]

    elbo_vars.var_G_s.v[] = E_G2_s_v - (E_G_s_v ^ 2)

    ############### only gradient and hessian code below
    (is_active_source && elbo_vars.elbo.has_gradient) || return

    var_G_s_d = elbo_vars.var_G_s.d
    E_G2_s_d = elbo_vars.E_G2_s.d
    E_G_s_d = elbo_vars.E_G_s.d

    @assert length(var_G_s_d) == length(E_G2_s_d) == length(E_G_s_d)

    @inbounds for ind1 = 1:length(var_G_s_d)
        var_G_s_d[ind1] = E_G2_s_d[ind1] - 2 * E_G_s_v * E_G_s_d[ind1]
    end

    ########## only hessian code below
    elbo_vars.elbo.has_hessian || return

    var_G_s_h = elbo_vars.var_G_s.h
    E_G2_s_h = elbo_vars.E_G2_s.h
    E_G_s_h = elbo_vars.E_G_s.h
    p1, p2 = size(var_G_s_h)
    @inbounds for ind2 = 1:p2, ind1 = 1:ind2
        var_G_s_h[ind1, ind2] =
            E_G2_s_h[ind1, ind2] - 2 * (
                E_G_s_v * E_G_s_h[ind1, ind2] +
                E_G_s_d[ind1, 1] * E_G_s_d[ind2, 1])
        var_G_s_h[ind2, ind1] = var_G_s_h[ind1, ind2]
    end
end


"""
Add the contributions from a single source at a single pixel to the
sensitive floast E_G and var_G, which are updated in place.
"""
function accumulate_source_pixel_brightness!{NumType <: Number}(
                    ea::ElboArgs,
                    vp::VariationalParams{NumType},
                    elbo_vars::ElboIntermediateVariables{NumType},
                    sb::SourceBrightness{NumType},
                    b::Int, s::Int,
                    is_active_source::Bool)
    calculate_E_G_s!(ea, vp, elbo_vars, sb, b, s, is_active_source)
    calculate_var_G_s!(ea, vp, elbo_vars, is_active_source)

    if is_active_source
        sa = findfirst(ea.active_sources, s)
        add_sources_sf!(elbo_vars.E_G, elbo_vars.E_G_s, sa)
        add_sources_sf!(elbo_vars.var_G, elbo_vars.var_G_s, sa)
    else
        # If the sources is inactive, simply accumulate the values.
        elbo_vars.E_G.v[] += elbo_vars.E_G_s.v[]
        elbo_vars.var_G.v[] += elbo_vars.var_G_s.v[]
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
                E_G::SensitiveFloat{NumType},
                var_G::SensitiveFloat{NumType},
                elbo::SensitiveFloat{NumType},
                x_nbm::AbstractFloat,
                iota::AbstractFloat)
    # See notes for a derivation. The log term is
    # log E[G] - Var(G) / (2 * E[G] ^2 )

    @inbounds begin
        E_G_v = E_G.v[]
        var_G_v = var_G.v[]

        # The gradients and Hessians are written as a f(x, y) = f(E_G2, E_G)
        log_term_value = log(E_G_v) - var_G_v / (2.0 * E_G_v ^ 2)

        # Add x_nbm * (log term * log(iota)) to the elbo.
        # If not calculating derivatives, add the values directly.
        elbo.v[] += x_nbm * (log(iota) + log_term_value)

        if elbo_vars.elbo.has_gradient
            elbo_vars.combine_grad[1] = -0.5 / (E_G_v ^ 2)
            elbo_vars.combine_grad[2] = 1 / E_G_v + var_G_v / (E_G_v ^ 3)

            if elbo_vars.elbo.has_hessian
                elbo_vars.combine_hess[1, 1] = 0.0
                elbo_vars.combine_hess[1, 2] = elbo_vars.combine_hess[2, 1] = 1 / E_G_v^3
                elbo_vars.combine_hess[2, 2] =
                    -(1 / E_G_v ^ 2 + 3 * var_G_v / (E_G_v ^ 4))
            end

            # Calculate the log term.
            combine_sfs!(
                var_G, E_G, elbo_vars.elbo_log_term,
                log_term_value, elbo_vars.combine_grad, elbo_vars.combine_hess)

            # Add to the ELBO.
            elbo_d = elbo.d
            elbo_vars_elbo_log_term_d = elbo_vars.elbo_log_term.d

            for ind in 1:length(elbo_d)
                elbo_d[ind] += x_nbm * elbo_vars_elbo_log_term_d[ind]
            end

            if elbo_vars.elbo.has_hessian
                elbo_h = elbo.h
                elbo_vars_elbo_log_term_h = elbo_vars.elbo_log_term.h
                for ind in 1:length(elbo_h)
                    elbo_h[ind] += x_nbm * elbo_vars_elbo_log_term_h[ind]
                end
            end
        end
    end
end


function add_pixel_term!{NumType <: Number}(
                    ea::ElboArgs,
                    vp::VariationalParams{NumType},
                    n::Int, h::Int, w::Int,
                    star_mcs::Array{BvnComponent{NumType}, 2},
                    gal_mcs::Array{GalaxyCacheComponent{NumType}, 4},
                    sbs::Vector{SourceBrightness{NumType}},
                    elbo_vars = ElboIntermediateVariables(NumType, ea.psf_K, ea.S, ea.Sa, true, false))
    img = ea.images[n]

    # This combines the sources into a single brightness value for the pixel.
    clear!(elbo_vars.E_G)
    clear!(elbo_vars.var_G)

    for s in 1:size(ea.patches, 1)
        p = ea.patches[s,n]

        h2 = h - p.bitmap_offset[1]
        w2 = w - p.bitmap_offset[2]

        hw = SVector{2,Float64}(h, w)
        is_active_source = s in ea.active_sources  # fast?

        populate_fsm!(elbo_vars.bvn_derivs,
                      elbo_vars.fs0m,
                      elbo_vars.fs1m,
                      s, hw, is_active_source,
                      p.wcs_jacobian,
                      gal_mcs, star_mcs)

        H2, W2 = size(p.active_pixel_bitmap)
        if 1 <= h2 <= H2 && 1 <= w2 < W2 && p.active_pixel_bitmap[h2, w2]
            is_active_source = s in ea.active_sources
            accumulate_source_pixel_brightness!(ea, vp, elbo_vars,
                sbs[s], ea.images[n].b, s, is_active_source)
        end
    end

    # There are no derivatives with respect to epsilon, so can safely add
    # to the value.
    elbo_vars.E_G.v[] += img.epsilon_mat[h, w]

    # Add the terms to the elbo given the brightness.
    add_elbo_log_term!(elbo_vars,
                       elbo_vars.E_G,
                       elbo_vars.var_G,
                       elbo_vars.elbo,
                       img.pixels[h,w],
                       img.iota_vec[h])
    add_scaled_sfs!(elbo_vars.elbo,
                    elbo_vars.E_G,
                    -img.iota_vec[h])

    # Subtract the log factorial term. This is not a function of the
    # parameters so the derivatives don't need to be updated. Note that
    # even though this does not affect the ELBO's maximum, it affects
    # the optimization convergence criterion, so I will leave it in for now.
    elbo_vars.elbo.v[] -= lfact(img.pixels[h,w])
end


"""
Return the expected log likelihood for all bands in a section
of the sky.
Returns: A sensitive float with the log likelihood.
"""
function elbo_likelihood{NumType <: Number}(
             ea::ElboArgs,
             vp::VariationalParams{NumType},
             elbo_vars = ElboIntermediateVariables(NumType, ea.psf_K, ea.S, ea.Sa))
    clear!(elbo_vars)

    # this call loops over light sources (but not images)
    sbs = load_source_brightnesses(ea, vp)

    for n in 1:ea.N
        img = ea.images[n]

        # could preallocate these---outside of elbo_likehood even to use for
        # all ~50 evalulations of the likelihood
        # This convolves the PSF with the star/galaxy model, returning a
        # mixture of bivariate normals.
        star_mcs, gal_mcs = Model.load_bvn_mixtures!(
                                    elbo_vars.star_mcs, elbo_vars.gal_mcs,
                                    ea.S, ea.patches,
                                    vp, ea.active_sources,
                                    ea.psf_K, n,
                                    elbo_vars.elbo.has_gradient,
                                    elbo_vars.elbo.has_hessian)

        # if there's only one active source, we know each pixel we visit
        # hasn't been visited before, so no need to allocate memory.
        # currently length(ea.active_sources) > 1 only in unit tests, never
        # when invoked from `bin`.
        already_visited = length(ea.active_sources) == 1 ?
                              falses(0, 0) :
                              falses(size(img.pixels))

        # iterate over the pixels by iterating over the patches, and visiting
        # all the pixels in the patch that are active and haven't already been
        # visited
        for s in ea.active_sources
            p = ea.patches[s, n]
            H2, W2 = size(p.active_pixel_bitmap)
            for w2 in 1:W2, h2 in 1:H2
                # (h2, w2) index the local patch, while (h, w) index the image
                h = p.bitmap_offset[1] + h2
                w = p.bitmap_offset[2] + w2

                if !p.active_pixel_bitmap[h2, w2]
                    continue
                end

                # if there's only one source to visit, we know this pixel is new
                if length(ea.active_sources) != 1
                    if already_visited[h,w]
                        continue
                    end
                    already_visited[h,w] = true
                end

                # Some pixels that are NaN in the original image may be active
                # for the convolution code.
                if isnan(img.pixels[h, w])
                    continue
                end

                # if we're here it's a unique active pixel.
                # Note that although we are iterating over pixels within a
                # single patch, add_pixel_term /also/ iterates over patches to
                # find all patches that overlap with this pixel.
                add_pixel_term!(ea, vp, n, h, w, star_mcs, gal_mcs, sbs, elbo_vars)
            end
        end
    end

    assert_all_finite(elbo_vars.elbo)
    elbo_vars.elbo
end


"""
Calculates and returns the ELBO and its derivatives for all the bands
of an image.
Returns: A sensitive float containing the ELBO for the image.
"""
function elbo{NumType <: Number}(
                 ea::ElboArgs,
                 vp::VariationalParams{NumType},
                 elbo_vars::ElboIntermediateVariables{NumType} =
                    ElboIntermediateVariables(NumType, ea.psf_K, ea.S, ea.Sa,
                                              true, NumType <: AbstractFloat),
                 kl_source = SensitiveFloat{NumType}(length(CanonicalParams), 1,
                       elbo_vars.elbo.has_gradient, elbo_vars.elbo.has_hessian),
                 kl_helper = KLDivergence.get_kl_helper(NumType))
    @assert(all([all(isfinite, vs) for vs in vp]), "vp contains NaNs or Infs")
    ret = elbo_likelihood(ea, vp, elbo_vars)
    if ea.include_kl
        KLDivergence.subtract_kl_all_sources!(ea, vp, ret, kl_source, kl_helper)
    end
    assert_all_finite(elbo_vars.elbo)
    return ret
end
