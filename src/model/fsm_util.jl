"""
Convolve the current locations and galaxy shapes with the PSF.  If
calculate_derivs is true, also calculate derivatives and hessians for
active sources.

Args:
 - S (formerly from ElboArgs)
 - patches (formerly from ElboArgs)
 - vp: (formerly from ElboArgs)
 - active_sources (formerly from ElboArgs)
 - psf: A vector of PSF components
 - b: The current band
 - calculate_derivs: Whether to calculate derivatives for active sources.
 - calculate_hessian

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
                    S::Int64,
                    patches::Matrix{SkyPatch},
                    vp::VariationalParams{NumType},
                    active_sources::Vector{Int},
                    psf_K::Int64,
                    b::Int;
                    calculate_derivs::Bool=true,
                    calculate_hessian::Bool=true)
    star_mcs = Array(BvnComponent{NumType}, psf_K, S)
    gal_mcs = Array(GalaxyCacheComponent{NumType}, psf_K, 8, 2, S)

    # TODO: do not keep any derviative information if the sources are not in
    # active_sources.
    for s in 1:S
        psf = patches[s, b].psf
        vs = vp[s]

        world_loc = vs[[ids.u[1], ids.u[2]]]
        m_pos = Model.linear_world_to_pix(patches[s, b].wcs_jacobian,
                                          patches[s, b].center,
                                          patches[s, b].pixel_center, world_loc)

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
                        calculate_derivs && (s in active_sources),
                        calculate_hessian)
                end
            end
        end
    end

    star_mcs, gal_mcs


    star_mcs = Array(BvnComponent{NumType}, psf_K, S)
    gal_mcs = Array(GalaxyCacheComponent{NumType}, psf_K, 8, 2, S)

    # TODO: do not keep any derviative information if the sources are not in
    # active_sources.
    for s in 1:S
        psf = patches[s, b].psf
        vs = vp[s]

        world_loc = vs[[ids.u[1], ids.u[2]]]
        m_pos = Model.linear_world_to_pix(patches[s, b].wcs_jacobian,
                                          patches[s, b].center,
                                          patches[s, b].pixel_center, world_loc)

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
                        calculate_derivs && (s in active_sources),
                        calculate_hessian)
                end
            end
        end
    end

    star_mcs, gal_mcs
end


type ModelIntermediateVariables{NumType <: Number}

    bvn_derivs::BivariateNormalDerivatives{NumType}

    # Vectors of star and galaxy bvn quantities from all sources for a pixel.
    # The vector has one element for each active source, in the same order
    # as ea.active_sources.

    # TODO: you can treat this the same way as E_G_s and not keep a vector around.
    fs0m_vec::Vector{SensitiveFloat{StarPosParams, NumType}}
    fs1m_vec::Vector{SensitiveFloat{GalaxyPosParams, NumType}}


    # Pre-allocated memory for the gradient and Hessian of combine functions.
    combine_grad::Vector{NumType}
    combine_hess::Matrix{NumType}

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
function ModelIntermediateVariables(NumType::DataType,
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

    combine_grad = zeros(NumType, 2)
    combine_hess = zeros(NumType, 2, 2)

    ModelIntermediateVariables{NumType}(
        bvn_derivs, fs0m_vec, fs1m_vec,
        combine_grad, combine_hess,
        calculate_derivs, calculate_hessian)
end


"""
Populate fs0m_vec and fs1m_vec for all sources for a given pixel.

Args:
    - model_vars: Model intermediate values.
    - patches: (formerly from the ElboArgs object)
    - active_sources: (formerly from the ElboArgs object)
    - tile_sources: A vector of integers of sources in 1:ea.S affecting the tile
    - tile: An ImageTile
    - h, w: The integer locations of the pixel within the tile
    - gal_mcs: Galaxy components
    - star_mcs: Star components

Returns:
    Updates model_vars.fs0m_vec and model_vars.fs1m_vec in place with the total
    shape contributions to this pixel's brightness.
"""
function populate_fsm_vecs!{NumType <: Number}(
                    bvn_derivs::BivariateNormalDerivatives{NumType},
                    fs0m_vec::Vector{SensitiveFloat{StarPosParams, NumType}},
                    fs1m_vec::Vector{SensitiveFloat{GalaxyPosParams, NumType}},
                    mv_calculate_derivs::Bool,
                    mv_calculate_hessian::Bool,
                    patches::Matrix{SkyPatch},
                    active_sources::Vector{Int},
                    psf_K::Int64,
                    num_allowed_sd::Float64,
                    tile_sources::Vector{Int},
                    tile::ImageTile,
                    h::Int, w::Int,
                    gal_mcs::Array{GalaxyCacheComponent{NumType}, 4},
                    star_mcs::Array{BvnComponent{NumType}, 2})

    x = Float64[tile.h_range[h], tile.w_range[w]]
    for s in tile_sources
        # ensure tile.b is a filter band, not an image's index
        @assert 1 <= tile.b <= B
        wcs_jacobian = patches[s, tile.b].wcs_jacobian
        active_source = s in active_sources

        calculate_hessian =
            mv_calculate_hessian && mv_calculate_derivs && active_source
        clear!(fs0m_vec[s], calculate_hessian)
        for k = 1:psf_K # PSF component
            if (num_allowed_sd == Inf ||
                check_point_close_to_bvn(star_mcs[k, s], x, num_allowed_sd))
                accum_star_pos!(
                    bvn_derivs, fs0m_vec,
                    mv_calculate_derivs,
                    mv_calculate_hessian,
                    s, star_mcs[k, s], x, wcs_jacobian, active_source)
            end
        end

        clear!(fs1m_vec[s], calculate_hessian)
        for i = 1:2 # Galaxy types
            for j in 1:8 # Galaxy component
                # If i == 2 then there are only six galaxy components.
                if (i == 1) || (j <= 6)
                    for k = 1:psf_K # PSF component
                        if (num_allowed_sd == Inf ||
                            check_point_close_to_bvn(
                                gal_mcs[k, j, i, s].bmc, x, num_allowed_sd))
                            accum_galaxy_pos!(
                                bvn_derivs, fs1m_vec,
                                mv_calculate_derivs,
                                mv_calculate_hessian,
                                s, gal_mcs[k, j, i, s], x, wcs_jacobian,
                                active_source)
                        end
                    end
                end
            end
        end
    end
end


function populate_fsm_vecs!{NumType <: Number}(
                    model_vars::ModelIntermediateVariables{NumType},
                    patches::Matrix{SkyPatch},
                    active_sources::Vector{Int},
                    psf_K::Int64,
                    num_allowed_sd::Float64,
                    tile_sources::Vector{Int},
                    tile::ImageTile,
                    h::Int, w::Int,
                    gal_mcs::Array{GalaxyCacheComponent{NumType}, 4},
                    star_mcs::Array{BvnComponent{NumType}, 2})
    Model.populate_fsm_vecs!(model_vars.bvn_derivs,
                             model_vars.fs0m_vec,
                             model_vars.fs1m_vec,
                             model_vars.calculate_derivs,
                             model_vars.calculate_hessian,
                             patches, active_sources, psf_K, num_allowed_sd,
                             tile_sources,
                             tile, h, w, gal_mcs, star_mcs)
end


"""
Add the contributions of a star's bivariate normal term to the ELBO,
by updating elbo_vars.fs0m_vec[s] in place.

Args:
    - bvn_derivs: (formerly from elbo_vars)
    - fs0m_vec: vector of sensitive floats, populated by this method
    - calculate_derivs: the and of active_source and formerly elbo_vars.calculate_derivs
    - calculate_hessian: elbo_vars: Elbo intermediate values.
    - s: The index of the current source in 1:S
    - bmc: The component to be added
    - x: An offset for the component in pixel coordinates (e.g. a pixel location)
    - wcs_jacobian: The jacobian of the function pixel = F(world) at this location.

Returns:
    Updates elbo_vars.fs0m_vec[s] in place.
"""
function accum_star_pos!{NumType <: Number}(
                    bvn_derivs::BivariateNormalDerivatives{NumType},
                    fs0m_vec::Vector{SensitiveFloat{StarPosParams, NumType}},
                    calculate_derivs::Bool,
                    calculate_hessian::Bool,
                    s::Int,
                    bmc::BvnComponent{NumType},
                    x::Vector{Float64},
                    wcs_jacobian::Array{Float64, 2},
                    is_active_source::Bool)
    eval_bvn_pdf!(bvn_derivs, bmc, x)

    if calculate_derivs && is_active_source
        get_bvn_derivs!(bvn_derivs, bmc, true, false)
    end

    fs0m = fs0m_vec[s]
    fs0m.v[1] += bvn_derivs.f_pre[1]

    if calculate_derivs && is_active_source
        transform_bvn_ux_derivs!(bvn_derivs, wcs_jacobian, calculate_hessian)
        bvn_u_d = bvn_derivs.bvn_u_d
        bvn_uu_h = bvn_derivs.bvn_uu_h

        # Accumulate the derivatives.
        for u_id in 1:2
            fs0m.d[star_ids.u[u_id]] += bvn_derivs.f_pre[1] * bvn_u_d[u_id]
        end

        if calculate_hessian
            # Hessian terms involving only the location parameters.
            # TODO: redundant term
            for u_id1 in 1:2, u_id2 in 1:2
                fs0m.h[star_ids.u[u_id1], star_ids.u[u_id2]] +=
                    bvn_derivs.f_pre[1] * (bvn_uu_h[u_id1, u_id2] +
                    bvn_u_d[u_id1] * bvn_u_d[u_id2])
            end
        end
    end
end


"""
Add the contributions of a galaxy component term to the ELBO by
updating fs1m in place.

Args:
    - s: The index of the current source in 1:S
    - gcc: The galaxy component to be added
    - x: An offset for the component in pixel coordinates (e.g. a pixel location)
    - wcs_jacobian: The jacobian of the function pixel = F(world) at this location.
    - calculate_derivs: Whether to calculate derivatives.

Returns:
    Updates elbo_vars.fs1m_vec[s] in place.
"""
function accum_galaxy_pos!{NumType <: Number}(
                    bvn_derivs::BivariateNormalDerivatives{NumType},
                    fs1m_vec::Vector{SensitiveFloat{GalaxyPosParams, NumType}},
                    calculate_derivs::Bool,
                    calculate_hessian::Bool,
                    s::Int,
                    gcc::GalaxyCacheComponent{NumType},
                    x::Vector{Float64},
                    wcs_jacobian::Array{Float64, 2},
                    is_active_source::Bool)
    eval_bvn_pdf!(bvn_derivs, gcc.bmc, x)
    f = bvn_derivs.f_pre[1] * gcc.e_dev_i
    fs1m = fs1m_vec[s]
    fs1m.v[1] += f

    if calculate_derivs && is_active_source

        get_bvn_derivs!(bvn_derivs, gcc.bmc,
            calculate_hessian, calculate_hessian)
        transform_bvn_derivs!(
            bvn_derivs, gcc.sig_sf, wcs_jacobian, calculate_hessian)

        bvn_u_d = bvn_derivs.bvn_u_d
        bvn_uu_h = bvn_derivs.bvn_uu_h
        bvn_s_d = bvn_derivs.bvn_s_d
        bvn_ss_h = bvn_derivs.bvn_ss_h
        bvn_us_h = bvn_derivs.bvn_us_h

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
        fs1m.d[gal_ids.e_dev] += gcc.e_dev_dir * bvn_derivs.f_pre[1]

        if calculate_hessian
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
                    bvn_derivs.f_pre[1] * gcc.e_dev_dir * bvn_u_d[u_id]
                fs1m.h[devi, ui] = fs1m.h[ui, devi]
            end
            for shape_id in 1:length(gal_shape_ids)
                si = gal_shape_alignment[shape_id]
                fs1m.h[si, devi] +=
                    bvn_derivs.f_pre[1] * gcc.e_dev_dir * bvn_s_d[shape_id]
                fs1m.h[devi, si] = fs1m.h[si, devi]
            end
        end # if calculate hessian
    end # if is_active_source

end
