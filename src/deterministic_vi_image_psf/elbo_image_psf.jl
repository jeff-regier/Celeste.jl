

# Load galaxy bivariate normals with no PSF.  Used in DeterministicVIImagePSF.
function load_gal_bvn_mixtures{NumType <: Number}(
                    S::Int64,
                    patches::Matrix{SkyPatch},
                    source_params::Vector{Vector{NumType}},
                    active_sources::Vector{Int},
                    n::Int;
                    calculate_gradient::Bool=true,
                    calculate_hessian::Bool=true)
    # To maintain consistency with the rest of the code, use a 4d
    # array.  The first dimension was previously the PSF component.
    gal_mcs = Array{GalaxyCacheComponent{NumType}}(1, 8, 2, S)

    for s in 1:S
        sp  = source_params[s]
        world_loc = sp[lidx.u]
        p = patches[s, n]
        m_pos = linear_world_to_pix(
            p.wcs_jacobian, p.center, p.pixel_center, world_loc)

        for i = 1:2 # i indexes dev vs exp galaxy types.
            e_dev_dir = (i == 1) ? 1. : -1.
            e_dev_i = (i == 1) ? sp[lidx.e_dev] : 1. - sp[lidx.e_dev]

            # Galaxies of type 1 have 8 components, and type 2 have 6 components.
            for j in 1:[8,6][i]
                gal_mcs[1, j, i, s] = GalaxyCacheComponent(
                    e_dev_dir, e_dev_i, galaxy_prototypes[i][j], m_pos,
                    sp[lidx.e_axis], sp[lidx.e_angle], sp[lidx.e_scale],
                    calculate_gradient && (s in active_sources),
                    calculate_hessian)
            end
        end
    end

    gal_mcs
end


# Get a GalaxyCacheComponent with no PSF.
import ..Model.GalaxyCacheComponent
function GalaxyCacheComponent{NumType <: Number}(
    e_dev_dir::Float64, e_dev_i::NumType,
    gc::GalaxyComponent, u::Vector{NumType},
    e_axis::NumType, e_angle::NumType, e_scale::NumType,
    calculate_gradient::Bool, calculate_hessian::Bool)

    # Declare in advance to save memory allocation.
    const empty_sig_sf =
        GalaxySigmaDerivs(Matrix{NumType}(0, 0), Array{NumType}(0, 0, 0))

    XiXi = get_bvn_cov(e_axis, e_angle, e_scale)
    var_s = gc.nuBar * XiXi

    # d siginv / dsigma is only necessary for the Hessian.
    bmc = BvnComponent{NumType}(
        SVector{2, NumType}(u), var_s, gc.etaBar,
        calculate_gradient && calculate_hessian)

    if calculate_gradient
        sig_sf = GalaxySigmaDerivs(
            e_angle, e_axis, e_scale, XiXi, calculate_hessian)
        sig_sf.j .*= gc.nuBar
        if calculate_hessian
            # The tensor is only needed for the Hessian.
            sig_sf.t .*= gc.nuBar
        end
    else
        sig_sf = empty_sig_sf
    end

    GalaxyCacheComponent(e_dev_dir, e_dev_i, bmc, sig_sf)
end



###############

const dir = "/home/rgiordan/Documents/git_repos/CelesteDev.jl/"


"""
Convolve a populated set of SensitiveFloat matrices in fsms with the PSF
and store them in the matching fs*m_conv SensitiveFloat matrices.
"""
function convolve_fs1m_image!(fsms::FSMSensitiveFloatMatrices, s::Int)
    for h in 1:size(fsms.fs1m_image, 1), w in 1:size(fsms.fs1m_image, 2)
        fsms.fs1m_image_padded[h, w] = fsms.fs1m_image[h, w]
    end

    convolve_sensitive_float_matrix!(
        fsms.fs1m_image_padded, fsms.psf_fft, fsms.fs1m_conv_padded)

    for h in 1:size(fsms.fs1m_image, 1), w in 1:size(fsms.fs1m_image, 2)
        fsms.fs1m_conv[h, w] =
            fsms.fs1m_conv_padded[fsms.pad_pix_h + h, fsms.pad_pix_w + w]
    end

    # Set return type
    return true
end


function clear_fs1m!(fsms::FSMSensitiveFloatMatrices)
    for sf in fsms.fs1m_image clear!(sf) end
    for sf in fsms.fs1m_image_padded clear!(sf) end
    for sf in fsms.fs1m_conv clear!(sf) end
    for sf in fsms.fs1m_conv_padded clear!(sf) end
end


function clear_brightness!(fsms::FSMSensitiveFloatMatrices)
    for sf in fsms.E_G clear!(sf) end
    for sf in fsms.var_G clear!(sf) end
end


"""
Populate the fs1m shape matrices and convolve with the PSF for a given
source and band.  Assumes that fsms.psf_fft has already been set.

The result is the sources shapes, convolved with the PSF, stored in fsms.fs1m_conv.

TODO: pass in derivative flags
"""
function populate_gal_fsm_image!(
            ea::ElboArgs{Float64},
            s::Int,
            n::Int,
            gal_mcs::Array{GalaxyCacheComponent{Float64}, 4},
            fsms::FSMSensitiveFloatMatrices)
    clear_fs1m!(fsms)
    is_active_source = s in ea.active_sources
    p = ea.patches[s, n]
    H_patch, W_patch = size(p.active_pixel_bitmap)
    for w_patch in 1:W_patch, h_patch in 1:H_patch
        if !p.active_pixel_bitmap[h_patch, w_patch]
            continue
        end
        h_image = h_patch + p.bitmap_offset[1]
        w_image = w_patch + p.bitmap_offset[2]

        h_fsm = h_image - fsms.h_lower + 1
        w_fsm = w_image - fsms.w_lower + 1

        x = SVector{2, Float64}([h_image, w_image])
        populate_gal_fsm!(ea.elbo_vars.bvn_derivs,
                          fsms.fs1m_image[h_fsm, w_fsm],
                          s, x, is_active_source,
                          p.wcs_jacobian,
                          gal_mcs)
    end
    convolve_fs1m_image!(fsms, s)
end


"""
Populate the fs1m shape matrices and convolve with the PSF for a given
source and band.  Assumes that fsms.psf_fft has already been set.

The result is the sources shapes, convolved with the PSF, stored in fsms.fs1m_conv.

TODO: pass in derivative flags
"""
function populate_star_fsm_image!(
            ea::ElboArgs{Float64},
            s::Int,
            n::Int,
            psf_image::Matrix{Float64},
            fs0m_conv::fs0mMatrix,
            h_lower::Int, w_lower::Int,
            kernel_fun::Any, # TODO: Is there a type that can be used here?
            kernel_width::Int)
    for sf in fs0m_conv clear!(sf) end
    # The pixel location of the star.
    star_loc_pix =
        linear_world_to_pix(ea.patches[s, n].wcs_jacobian,
                            ea.patches[s, n].center,
                            ea.patches[s, n].pixel_center,
                            ea.vp[s][lidx.u]) -
        Float64[ h_lower - 1, w_lower - 1]
    interpolate!(kernel_fun, kernel_width,
                 fs0m_conv, psf_image, star_loc_pix,
                 ea.patches[s, n].wcs_jacobian,
                 ea.elbo_vars.elbo.has_gradient,
                 ea.elbo_vars.elbo.has_hessian)
end


# TODO: rename this to source image brighntess or something.
"""
Updates fsms.E_G and fsms.var_G in place with the contributions from this
source in this band.
"""
function accumulate_source_image_brightness!(
    ea::ElboArgs{Float64},
    s::Int,
    n::Int,
    fsms::FSMSensitiveFloatMatrices,
    sb::SourceBrightness{Float64})

    is_active_source = s in ea.active_sources
    calculate_hessian =
        ea.elbo_vars.elbo.has_hessian && ea.elbo_vars.elbo.has_gradient &&
        is_active_source

    image_fft = [ sf.v[] for sf in fsms.fs1m_conv ]
    p = ea.patches[s, n]
    H_patch, W_patch = size(p.active_pixel_bitmap)
    for w_patch in 1:W_patch, h_patch in 1:H_patch
        h_fsm = h_patch + p.bitmap_offset[1] - fsms.h_lower + 1
        w_fsm = w_patch + p.bitmap_offset[2] - fsms.w_lower + 1
        if !p.active_pixel_bitmap[h_patch, w_patch]
            continue
        end
        accumulate_source_pixel_brightness!(
                            ea.elbo_vars,
                            ea,
                            fsms.E_G[h_fsm, w_fsm],
                            fsms.var_G[h_fsm, w_fsm],
                            fsms.fs0m_conv[h_fsm, w_fsm],
                            fsms.fs1m_conv[h_fsm, w_fsm],
                            sb, ea.images[n].b, s, is_active_source)
    end
end


pixel_counter = 0

"""
Uses the values in fsms to add the contribution from this band to the ELBO.
"""
function accumulate_band_in_elbo!(
    ea::ElboArgs{Float64},
    fsm_mat::Matrix{FSMSensitiveFloatMatrices},
    sbs::Vector{SourceBrightness{Float64}},
    gal_mcs::Array{GalaxyCacheComponent{Float64}, 4},
    n::Int)

    for s in 1:ea.S
        # some sources don't appear in some images
        sum(ea.patches[s, n].active_pixel_bitmap) > 0 || continue

        fsms = fsm_mat[s, n]
        clear_brightness!(fsms)
        populate_star_fsm_image!(
            ea, s, n, fsms.psf, fsms.fs0m_conv,
            fsms.h_lower, fsms.w_lower,
            fsms.kernel_fun, fsms.kernel_width)
        populate_gal_fsm_image!(ea, s, n, gal_mcs, fsms)
        accumulate_source_image_brightness!(ea, s, n, fsms, sbs[s])
    end

    # Iterate only over active sources, since we have already added the
    # contributions from non-active sources to E_G and var_G.
    for s in ea.active_sources
        p = ea.patches[s, n]
        fsms = fsm_mat[s, n]
        H_patch, W_patch = size(p.active_pixel_bitmap)
        for w_patch in 1:W_patch, h_patch in 1:H_patch
            if !p.active_pixel_bitmap[h_patch, w_patch]
                continue
            end
            global pixel_counter += 1
            h_image = h_patch + p.bitmap_offset[1]
            w_image = w_patch + p.bitmap_offset[2]

            image = ea.images[n]
            this_pixel = image.pixels[h_image, w_image]

            if Base.isnan(this_pixel)
                continue
            end

            # These are indices within the fs?m image.
            h_fsm = h_image - fsms.h_lower + 1
            w_fsm = w_image - fsms.w_lower + 1

            E_G = fsms.E_G[h_fsm, w_fsm]
            var_G = fsms.var_G[h_fsm, w_fsm]

            # There are no derivatives with respect to epsilon, so can
            # afely add to the value.
            E_G.v[] += image.epsilon_mat[h_image, w_image]

            if E_G.v[] < 0
                # warn("Image ", n, " sources ", s, " pixel ", (h_image, w_image),
                #      " has negative brightness ", E_G.v[])
                continue
            end

            # Note that with a kernel_width > 1 negative values are
            # possible, and this will result in an error in
            # add_elbo_log_term.

            # Add the terms to the elbo given the brightness.
            iota = image.iota_vec[h_image]
            add_elbo_log_term!(
                ea.elbo_vars, E_G, var_G, ea.elbo_vars.elbo, this_pixel, iota)
            add_scaled_sfs!(ea.elbo_vars.elbo, E_G, -iota)

            # Subtract the log factorial term. This is not a function of the
            # parameters so the derivatives don't need to be updated. Note
            # that even though this does not affect the ELBO's maximum,
            # it affects the optimization convergence criterion, so I will
            # leave it in for now.
            ea.elbo_vars.elbo.v[] -= lfact(this_pixel)
        end
    end
end


function elbo_likelihood_with_fft!(
    ea::ElboArgs,
    fsm_mat::Matrix{FSMSensitiveFloatMatrices})

    sbs = load_source_brightnesses(ea,
        calculate_gradient=ea.elbo_vars.elbo.has_gradient,
        calculate_hessian=ea.elbo_vars.elbo.has_hessian)

    clear!(ea.elbo_vars.elbo)
    for n in 1:ea.N
        gal_mcs = load_gal_bvn_mixtures(
                ea.S, ea.patches, ea.vp, ea.active_sources, n,
                calculate_gradient=ea.elbo_vars.elbo.has_gradient,
                calculate_hessian=ea.elbo_vars.elbo.has_hessian)
        accumulate_band_in_elbo!(ea, fsm_mat, sbs, gal_mcs, n)
    end
end


function load_fsm_mat(ea::ElboArgs,
                      images;
                      use_raw_psf=true,
                      use_trimmed_psf=true)
    if use_raw_psf
        psf_image_mat = Matrix{Matrix{Float64}}(ea.S, ea.N)
        for n in 1:ea.N, s in 1:ea.S
            img = images[n]
            world_loc = ea.vp[s][lidx.u]
            pixel_loc = WCS.world_to_pix(img.wcs, world_loc)
            psf_image_mat[s, n] =
                eval_psf(img.raw_psf_comp, pixel_loc[1], pixel_loc[2])
        end
    else
        psf_image_mat = Matrix{Float64}[
            get_psf_at_point(ea.patches[s, b].psf) for s in 1:ea.S, b in 1:ea.N]
    end

    if use_trimmed_psf
        for n in 1:ea.N, s in 1:ea.S
            psf_image_mat[s, n] = trim_psf(psf_image_mat[s, n])
        end
    end

    fsm_mat = FSMSensitiveFloatMatrices[
        FSMSensitiveFloatMatrices() for s in 1:ea.S, n in 1:ea.N]
    initialize_fsm_sf_matrices!(fsm_mat, ea, psf_image_mat)
    fsm_mat
end

function initialize_fft_elbo_parameters(
    images::Vector{Image},
    vp::VariationalParams{Float64},
    patches::Matrix{SkyPatch},
    active_sources::Vector{Int};
    use_raw_psf=true,
    use_trimmed_psf=true,
    allocate_fsm_mat=true,
    min_radius_pix=Nullable{Float64}())

    ea = ElboArgs(images, vp, patches, active_sources, psf_K=1)
    load_active_pixels!(images, ea.patches; exclude_nan=false, min_radius_pix=min_radius_pix)

    fsm_mat = nothing
    if allocate_fsm_mat
        fsm_mat = load_fsm_mat(ea, images; use_raw_psf=use_raw_psf, use_trimmed_psf=use_trimmed_psf)
    end

    ea, fsm_mat
end


@doc """
Return a function callback for an FFT elbo.
"""
function get_fft_elbo_function{T}(
    ea::ElboArgs{T}, fsm_mat::Matrix{FSMSensitiveFloatMatrices})
    function elbo_fft_opt(ea::ElboArgs)
        @assert ea.psf_K == 1
        elbo = ea.elbo_vars.elbo
        kl_source = SensitiveFloat{T}(length(CanonicalParams), 1,
                                      elbo.has_gradient, elbo.has_hessian)
        elbo_likelihood_with_fft!(ea, fsm_mat)
        kl_helper = KLDivergence.KL_HELPER_POOL[Base.Threads.threadid()]
        KLDivergence.subtract_kl_all_sources!(ea, elbo, kl_source, kl_helper)
        return deepcopy(elbo)
    end
end
