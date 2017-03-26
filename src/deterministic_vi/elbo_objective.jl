using Celeste: @syntactic_unroll, @implicit_transpose, is_implicitly_symmetric, fixup, Const
using Celeste.Model: Star, Galaxy, a_param, shape_params, brightness_params, bright_ids, ids_2_to_ids,
  dense_block_mapping, dense_blocks

"""
Calculate the contributions of a single source for a single pixel to
the sensitive floats E_G_s and var_G_s, which are cleared and updated in place.

Args:
    - ea: Model parameters
    - vp: the variational parameters
    - elbo_vars: Elbo intermediate values, with updated fs1m and fs0m.
    - sb: Source brightnesse
    - s: The source, in 1:ea.S
    - b: The band

Returns:
    Updates E_G_s, E_G2_s, and var_G_s in place with the brightness
    for this source at this pixel.
"""
function calculate_G_s!{NumType <: Number}(
                    vp::VariationalParams{NumType},
                    elbo_vars::ElboIntermediateVariables{NumType},
                    sb::SourceBrightness{NumType},
                    b::Int,
                    s::Int,
                    is_active_source::Bool)
    E_G_s = elbo_vars.E_G_s
    E_G2_s = elbo_vars.E_G2_s

    # The compiler can fold this into the arith below. No actual
    # time is spent here.
    clear!(E_G_s)
    clear!(E_G2_s)

    # Calculate E_G_s and E_G2_s
    @inbounds begin
        @syntactic_unroll for kind in (Star(), Galaxy())
            fsm_i = (kind == Star()) ? elbo_vars.fs0m : elbo_vars.fs1m

            i = (kind == Star()) ? 1 : 2
            sb_E_l_a_b_i = sb.E_l_a[i][b]
            sb_E_ll_a_b_i = sb.E_ll_a[i][b]
            sb_E_l_a_b_i_d = sb_E_l_a_b_i.d
            sb_E_ll_a_b_i_d = sb_E_ll_a_b_i.d

            sb_E_l_a_b_i_v = sb_E_l_a_b_i.v[]
            sb_E_ll_a_b_i_v = sb_E_ll_a_b_i.v[]

            fsm_i_v = fsm_i.v[]
            lf = sb_E_l_a_b_i_v * fsm_i_v
            llff = sb_E_ll_a_b_i_v * fsm_i_v^2

            a_i = vp[s][ids.a[i]]

            # Values
            E_G_s.v[] += a_i * lf
            E_G2_s.v[] += a_i * llff

            # Gradients
            (is_active_source && elbo_vars.elbo.has_gradient) || continue
            E_G_s.d[a_param(kind)] += lf
            E_G2_s.d[a_param(kind)] += llff

            E_G_s.d[shape_params(kind)] += (sb_E_l_a_b_i_v * a_i) * fsm_i.d[shape_params(kind)]
            E_G2_s.d[shape_params(kind)] += (sb_E_ll_a_b_i_v * 2 * fsm_i_v * a_i) * fsm_i.d[shape_params(kind)]

            E_G_s.d[brightness_params(kind)] += (a_i * fsm_i_v) * sb_E_l_a_b_i_d[brightness_params(kind)]
            E_G2_s.d[brightness_params(kind)] += (a_i * fsm_i_v^2) * sb_E_ll_a_b_i_d[brightness_params(kind)]

            # Hessians
            elbo_vars.elbo.has_hessian || continue
            E_G_s.h[brightness_params(kind), brightness_params(kind)] = (a_i * fsm_i_v) * sb_E_l_a_b_i.h[brightness_params(kind), brightness_params(kind)]
            E_G2_s.h[brightness_params(kind), brightness_params(kind)] = (a_i * fsm_i_v^2) * sb_E_ll_a_b_i.h[brightness_params(kind), brightness_params(kind)]

            # The u_u submatrix is shared between stars/galaxies, so use += here
            E_G_s.h[shape_params(kind), shape_params(kind)] += (a_i * sb_E_l_a_b_i_v) * fsm_i.h[shape_params(kind), shape_params(kind)]
            E_G2_s.h[shape_params(kind), shape_params(kind)] +=
                (2 * a_i * sb_E_ll_a_b_i_v) * (fsm_i_v * fsm_i.h[shape_params(kind), shape_params(kind)] + fsm_i.d[shape_params(kind)]*fsm_i.d[shape_params(kind)]')

            @implicit_transpose begin
                E_G_s.h[brightness_params(kind), a_param(kind)] = fsm_i_v * sb_E_l_a_b_i_d[brightness_params(kind)]
                E_G2_s.h[brightness_params(kind), a_param(kind)] = (fsm_i_v ^ 2) * sb_E_ll_a_b_i_d[brightness_params(kind)]

                E_G_s.h[shape_params(kind), a_param(kind)] = sb_E_l_a_b_i_v * fsm_i.d[shape_params(kind)]
                E_G2_s.h[shape_params(kind), a_param(kind)] = (sb_E_ll_a_b_i_v * 2 * fsm_i_v) * fsm_i.d[shape_params(kind)]

                E_G_s.h[brightness_params(kind), shape_params(kind)] = a_i * (sb_E_l_a_b_i_d[brightness_params(kind)] * fsm_i.d[shape_params(kind)]')
                E_G2_s.h[brightness_params(kind), shape_params(kind)] = (2 * a_i * fsm_i_v) * (sb_E_ll_a_b_i_d[brightness_params(kind)] * fsm_i.d[shape_params(kind)]')
            end
        end
    end
    nothing
end

# This is a hack, where we manually write things out
@eval function add_var_G_s!{NumType <: Number}(
                    reparametrized_E_G_d,
                    var_G::SensitiveFloat{NumType},
                    E_G_s::SingleSourceSensitiveFloat{NumType, CanonicalParams2},
                    E_G2_s::SingleSourceSensitiveFloat{NumType, CanonicalParams2},
                    s::Int)
                    
    # Calculate var_G
    var_G.v[] += E_G2_s.v[] - (E_G_s.v[] ^ 2)

    var_G.has_gradient || return

    P = length(CanonicalParams2)
    @assert P == var_G.local_P
    P_shifted = P * (s - 1)

    reparametrized_E_G_d[:] = E_G_s.d[ids_2_to_ids]
    reparametrized_E_G2_d = E_G2_s.d[ids_2_to_ids]
    for i = 1:P
        @fastmath var_G.d[P_shifted + i] += reparametrized_E_G2_d[i] - 2 * E_G_s.v[] * reparametrized_E_G_d[i]
    end
    
    var_G.has_hessian || return
    
    @inbounds begin
        # We do this in two steps. First we add the dense terms (E_G(2)_s), then
        # the sparse components.
        for i = 1:P
            @unroll_loop for j = 1:P
                @fastmath var_G.h[P_shifted + j, P_shifted + i] -= 2 * reparametrized_E_G_d[i] * reparametrized_E_G_d[j]
            end
        end
            
        @syntactic_unroll for (lhs, rhs) in $(zip(dense_block_mapping, dense_blocks))
            var_G.h[P_shifted + lhs[1], P_shifted + lhs[2]] += E_G2_s.h[rhs[1], rhs[2]] - 2 * E_G_s.v[] * E_G_s.h[rhs[1], rhs[2]]
        end
    end
end

@eval function add_var_G_s!{NumType <: Number}(
                    reparametrized_E_G_d,
                    var_G::SingleSourceSensitiveFloat{NumType, CanonicalParams},
                    E_G_s::SingleSourceSensitiveFloat{NumType, CanonicalParams2},
                    E_G2_s::SingleSourceSensitiveFloat{NumType, CanonicalParams2})
                    
    # Calculate var_G
    var_G.v[] += E_G2_s.v[] - (E_G_s.v[] ^ 2)

    var_G.has_gradient || return

    P = length(CanonicalParams2)

    @aliasscope @inbounds begin
        reparametrized_E_G_d[:] = E_G_s.d[ids_2_to_ids]
        reparametrized_E_G2_d = E_G2_s.d[ids_2_to_ids]
    end
    @aliasscope begin
        @inbounds for i = 1:P
            @fastmath var_G.d[i] += reparametrized_E_G2_d[i] - 2 * E_G_s.v[] * Const(reparametrized_E_G_d)[i]
        end
        
        var_G.has_hessian || return
        
        @inbounds begin
            # We do this in two steps. First we add the dense terms (E_G(2)_s), then
            # the sparse components.
            for i = 1:P
                @unroll_loop for j = 1:P
                    @fastmath var_G.h[j,i] -= 2 * Const(reparametrized_E_G_d)[i] * Const(reparametrized_E_G_d)[j]
                end
            end
                
            @syntactic_unroll for (lhs, rhs) in $(zip(dense_block_mapping, dense_blocks))
                var_G.h[lhs[1], lhs[2]] += E_G2_s.h[rhs[1], rhs[2]] - 2 * E_G_s.v[] * E_G_s.h[rhs[1], rhs[2]]
            end
        end
    end
end

@eval function SensitiveFloats.add_sources_sf!{NumType <: Number}(
                    sf_all::SSparseSensitiveFloat{NumType, CanonicalParams},
                    sf_s::SingleSourceSensitiveFloat{NumType, CanonicalParams2},
                    s::Int)
    sf_all.v[] += sf_s.v[]

    @assert size(sf_all.d[1], 1) == size(sf_s.d, 1)

    P = length(CanonicalParams2)

    @inbounds if sf_all.has_gradient
        reparam_sf_s_d = sf_s.d[ids_2_to_ids]
        for i = 1:P
            sf_all.d[s][i] += reparam_sf_s_d[i]
        end
    end

    @inbounds if sf_all.has_hessian
        @syntactic_unroll for (lhs, rhs) in $(zip(dense_block_mapping, dense_blocks))
            sf_all.h[s][lhs[1], lhs[2]] += fixup(sf_s.h[rhs[1], rhs[2]])
        end
    end            
end

@eval function SensitiveFloats.add_sources_sf!{NumType <: Number}(
                    sf_all::SensitiveFloat{NumType},
                    sf_s::SingleSourceSensitiveFloat{NumType, CanonicalParams2},
                    s::Int)
    sf_all.v[] += sf_s.v[]

    @assert size(sf_all.d, 1) == size(sf_s.d, 1)

    P = length(CanonicalParams2)
    @assert P == sf_all.local_P
    P_shifted = P * (s - 1)

    @inbounds if sf_all.has_gradient
        reparam_sf_s_d = sf_s.d[ids_2_to_ids]
        for i = 1:P
            sf_all.d[P_shifted + i] += reparam_sf_s_d[i]
        end
    end

    @inbounds if sf_all.has_hessian
        @syntactic_unroll for (lhs, rhs) in $(zip(dense_block_mapping, dense_blocks))
            sf_all.h[P_shifted + lhs[1], P_shifted + lhs[2]] += fixup(sf_s.h[rhs[1], rhs[2]])
        end
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
    calculate_G_s!(vp, elbo_vars, sb, b, s, is_active_source)

    if is_active_source
        sa = findfirst(ea.active_sources, s)
        add_sources_sf!(elbo_vars.E_G, elbo_vars.E_G_s, sa)
        add_var_G_s!(elbo_vars.reparametrized_E_G_d,
          elbo_vars.var_G[SensitiveFloats.Source(sa)],
          elbo_vars.E_G_s, elbo_vars.E_G2_s)
    else
        # If the sources is inactive, simply accumulate the values.
        elbo_vars.E_G.v[] += elbo_vars.E_G_s.v[]
        elbo_vars.var_G.v[] += elbo_vars.E_G2_s.v[] - (elbo_vars.E_G_s.v[] ^ 2)
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
                E_G::AbstractSensitiveFloat{NumType},
                var_G::AbstractSensitiveFloat{NumType},
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
        elbo.v[] += x_nbm * log(iota)
        
        if !elbo_vars.elbo.has_gradient
            elbo.v[] += x_nbm * log_term_value
        end

        if elbo_vars.elbo.has_gradient
            combine_grad = SVector{2, NumType}(
              -0.5 / (E_G_v ^ 2),
              1 / E_G_v + var_G_v / (E_G_v ^ 3))

            combine_hess = SMatrix{2, 2, NumType, 4}(
                 (0.0,1 / E_G_v^3,1 / E_G_v^3,
                -(1 / E_G_v ^ 2 + 3 * var_G_v / (E_G_v ^ 4))))
                
            # Calculate the log term.
            combine_sfs!(
                (oldval, newval)->oldval + x_nbm*newval,
                Const,
                var_G, E_G, elbo,
                log_term_value, combine_grad, combine_hess)
        end
    end
end


function add_pixel_term!{NumType <: Number}(
                    ea::ElboArgs,
                    vp::VariationalParams{NumType},
                    n::Int, h::Int, w::Int,
                    bvn_bundle::BvnBundle{NumType},
                    sbs::Vector{SourceBrightness{NumType}},
                    elbo_vars::ElboIntermediateVariables = ElboIntermediateVariables(NumType, ea.Sa))
    img = ea.images[n]

    clear!(elbo_vars.E_G)
    clear!(elbo_vars.var_G)

    for s in 1:ea.S
        p = ea.patches[s,n]

        h2 = h - p.bitmap_offset[1]
        w2 = w - p.bitmap_offset[2]

        H2, W2 = size(p.active_pixel_bitmap)
        if 1 <= h2 <= H2 && 1 <= w2 < W2 && p.active_pixel_bitmap[h2, w2]
            is_active_source = s in ea.active_sources

            # this if/else block is for reporting purposes only
            if is_active_source
                elbo_vars.active_pixel_counter[] += 1
            else
                elbo_vars.inactive_pixel_counter[] += 1
            end

            populate_fsm!(bvn_bundle.bvn_derivs,
                          elbo_vars.fs0m,
                          elbo_vars.fs1m,
                          s,
                          SVector{2,Float64}(h, w),
                          is_active_source,
                          p.wcs_jacobian,
                          bvn_bundle.gal_mcs,
                          bvn_bundle.star_mcs)

            accumulate_source_pixel_brightness!(ea, vp, elbo_vars,
                sbs[s], ea.images[n].b, s, is_active_source)
        end
    end

    # There are no derivatives with respect to epsilon, so can safely add
    # to the value.
    elbo_vars.E_G.v[] += img.sky[h, w]

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
    elbo_vars.elbo.v[] -= lgamma(img.pixels[h,w] + 1.0)
end


"""
Return the expected log likelihood for all bands in a section
of the sky.
Returns: A sensitive float with the log likelihood.
"""
function elbo_likelihood{T}(ea::ElboArgs,
                            vp::VariationalParams{T},
                            elbo_vars::ElboIntermediateVariables = ElboIntermediateVariables(T, ea.Sa),
                            bvn_bundle::BvnBundle{T} = BvnBundle{T}(ea.psf_K, ea.S))
    clear!(elbo_vars)
    clear!(bvn_bundle)

    # this call loops over light sources (but not images)
    sbs = load_source_brightnesses(ea, vp)

    for n in 1:ea.N
        img = ea.images[n]

        # could preallocate these---outside of elbo_likehood even to use for
        # all ~50 evalulations of the likelihood
        # This convolves the PSF with the star/galaxy model, returning a
        # mixture of bivariate normals.
        star_mcs, gal_mcs = Model.load_bvn_mixtures!(
                                    bvn_bundle.star_mcs,
                                    bvn_bundle.gal_mcs,
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
                add_pixel_term!(ea, vp, n, h, w, bvn_bundle, sbs, elbo_vars)
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
function elbo{T}(ea::ElboArgs,
                 vp::VariationalParams{T},
                 elbo_vars::ElboIntermediateVariables =
                        ElboIntermediateVariables(T, ea.Sa, true,  T<:AbstractFloat),
                 bvn_bundle::BvnBundle = BvnBundle{T}(ea.psf_K, ea.S))
    @assert(all(all(isfinite, vs) for vs in vp), "vp contains NaNs or Infs")
    result = elbo_likelihood(ea, vp, elbo_vars, bvn_bundle)
    ea.include_kl && KLDivergence.subtract_kl_all_sources!(ea, vp, result)
    assert_all_finite(elbo_vars.elbo)
    return result
end
