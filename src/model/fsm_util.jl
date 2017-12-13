
"""
The convolution of a one galaxy component with one PSF component.
It also contains the derivatives of sigma with respect to the shape parameters.
It does not contain the derivatives with respect to other parameters
(pos and gal_frac_dev) because they have easy expressions in terms of other known
quantities.

Args:
 - gal_frac_dev_dir: "Theta direction": this is 1 or -1, depending on whether
     increasing gal_frac_dev increases the weight of this GalaxyCacheComponent
     (1) or decreases it (-1).
 - gal_frac_dev_i: The weight given to this type of galaxy for this celestial object.
     This is either gal_frac_dev or (1 - gal_frac_dev).
 - gc: The galaxy component to be convolved
 - pc: The psf component to be convolved
 - pos: The location of the celestial object in pixel coordinates as a 2x1 vector
 - gal_axis_ratio: The ratio of the galaxy minor axis to major axis (0 < gal_axis_ratio <= 1)
 - gal_radius_px: The scale of the galaxy major axis

Attributes:
 - gal_frac_dev_dir: Same as input
 - gal_frac_dev_i: Same as input
 - bmc: A BvnComponent with the convolution.
 - dSigma: A 3x3 matrix containing the derivates of
     [Sigma11, Sigma12, Sigma22] (in the rows) with respect to
     [gal_axis_ratio, gal_angle, gal_radius_px] (in the columns)
"""
struct GalaxyCacheComponent{T<:Number}
    gal_frac_dev_dir::Float64
    gal_frac_dev_i::T
    bmc::BvnComponent{T}
    sig_sf::GalaxySigmaDerivs{T}
    # [Sigma11, Sigma12, Sigma22] x [gal_axis_ratio, gal_angle, gal_radius_px]
end

function GalaxyCacheComponent(
        gal_frac_dev_dir::Float64,
        gal_frac_dev_i::T,
        gc::GalaxyComponent,
        pc::PsfComponent,
        pos::Vector{T},
        gal_axis_ratio::T,
        gal_angle::T,
        gal_radius_px::T,
        calculate_gradient::Bool,
        calculate_hessian::Bool) where {T<:Number}

    XiXi = get_bvn_cov(gal_axis_ratio, gal_angle, gal_radius_px)
    mean_s = @SVector T[pc.xiBar[1] + pos[1], pc.xiBar[2] + pos[2]]
    var_s = pc.tauBar + gc.nuBar * XiXi
    weight = pc.alphaBar * gc.etaBar  # excludes gal_frac_dev

    # d siginv / dsigma is only necessary for the Hessian.
    bmc = BvnComponent(mean_s, var_s, weight, calculate_gradient && calculate_hessian)

    if calculate_gradient
        sig_sf = GalaxySigmaDerivs(gal_angle, gal_axis_ratio, gal_radius_px, XiXi,
                                   gc.nuBar, calculate_hessian)
    else
        sig_sf = GalaxySigmaDerivs(T)
    end

    GalaxyCacheComponent(gal_frac_dev_dir, gal_frac_dev_i, bmc, sig_sf)
end


struct BvnBundle{T<:Real}
    bvn_derivs::BivariateNormalDerivatives{T}
    star_mcs::Matrix{BvnComponent{T}}
    gal_mcs::Array{GalaxyCacheComponent{T},4}
    function (::Type{BvnBundle{T}}){T}(psf_K::Int, S::Int)
        return new{T}(BivariateNormalDerivatives{T}(),
                      Matrix{BvnComponent{T}}(psf_K, S),
                      Array{GalaxyCacheComponent{T}}(psf_K, 8, 2, S))
    end
end

function zero!(bvn_bundle::BvnBundle)
    BivariateNormals.zero!(bvn_bundle.bvn_derivs)
    return bvn_bundle
end


"""
Convolve the current locations and galaxy shapes with the PSF.  If
calculate_gradient is true, also calculate derivatives and hessians for
active sources.

Args:
 - S (formerly from ElboArgs)
 - patches (formerly from ElboArgs)
 - vp: (formerly from ElboArgs)
 - active_sources (formerly from ElboArgs)
 - psf_K: Number of psf components (psf from patches object)
 - n: the image id (not the band)
 - calculate_gradient: Whether to calculate derivatives for active sources.
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
function load_bvn_mixtures!(
                    #outputs
                    star_mcs::Matrix{BvnComponent{T}},
                    gal_mcs::Array{GalaxyCacheComponent{T},4},
                    #inputs
                    S::Int64,
                    patches::Matrix{ImagePatch},
                    source_params::Vector{Vector{T}},
                    active_sources::Vector{Int},
                    psf_K::Int64,
                    n::Int,
                    calculate_gradient::Bool=true,
                    calculate_hessian::Bool=true) where {T<:Number}
    @assert size(star_mcs, 1) == psf_K
    @assert size(star_mcs, 2) == S
    # TODO: do not keep any derviative information if the sources are not in
    # active_sources.

    for s in 1:S
        psf = patches[s, n].psf
        sp  = source_params[s]

        # TODO: it's a lucky coincidence that lidx.pos = ids.pos.
        # That's why this works when called from both log_prob.jl with `sp`
        # and elbo_objective.jl with `vp`.
        # We need a safer way to let both methods call this method.
        world_loc = sp[lidx.pos]
        m_pos = Model.linear_world_to_pix(patches[s, n].wcs_jacobian,
                                          patches[s, n].world_center,
                                          patches[s, n].pixel_center, world_loc)

        # Convolve the star locations with the PSF.
        for k in 1:psf_K
            pc = psf[k]
            mean_s = @SVector T[pc.xiBar[1] + m_pos[1], pc.xiBar[2] + m_pos[2]]
            star_mcs[k, s] = BvnComponent(mean_s, pc.tauBar, pc.alphaBar, false)
        end

        # Convolve the galaxy representations with the PSF.
        for i = 1:2 # i indexes dev vs exp galaxy types.
            gal_frac_dev_dir = (i == 1) ? 1. : -1.
            gal_frac_dev_i = (i == 1) ? sp[lidx.gal_frac_dev] : 1. - sp[lidx.gal_frac_dev]

            # Galaxies of type 1 have 8 components, and type 2 have 6 components.
            for j in 1:ifelse(i == 1, 8, 6)
                for k = 1:psf_K
                    gal_mcs[k, j, i, s] = GalaxyCacheComponent(
                        gal_frac_dev_dir, gal_frac_dev_i, galaxy_prototypes[i][j], psf[k],
                        m_pos,
                        sp[lidx.gal_axis_ratio], sp[lidx.gal_angle], sp[lidx.gal_radius_px],
                        calculate_gradient && (s in active_sources),
                        calculate_hessian)
                end
            end
        end
    end

    star_mcs, gal_mcs
end

function load_bvn_mixtures(
        S::Int64,
        patches::Matrix{ImagePatch},
        source_params::Vector{Vector{T}},
        active_sources::Vector{Int},
        psf_K::Int64,
        n::Int;
        calculate_gradient::Bool=true,
        calculate_hessian::Bool=true) where {T<:Number}
    star_mcs = Matrix{BvnComponent{T}}(psf_K, S)
    gal_mcs  = Array{GalaxyCacheComponent{T}}(psf_K, 8, 2, S)
    load_bvn_mixtures!(star_mcs, gal_mcs, S, patches, source_params,
                       active_sources, psf_K, n, calculate_gradient,
                       calculate_hessian)
end


"""
Populate fs0m and fs1m for source s in the a given pixel.

Non-standard args:
    - x: The pixel location in the image
"""
function populate_gal_fsm!(
        fs1m::SensitiveFloat{T},
        bvn_derivs::BivariateNormalDerivatives{T},
        s::Int,
        h::Int64,
        w::Int64,
        is_active_source::Bool,
        wcs_jacobian::Matrix{Float64},
        gal_mcs::Array{GalaxyCacheComponent{T}, 4}) where {T<:Number}
    SensitiveFloats.zero!(fs1m)
    x = SVector{2,Float64}(h, w)

    for i = 1:2 # Galaxy types
        for j in 1:8 # Galaxy component
            # If i == 2 then there are only six galaxy components.
            if (i == 1) || (j <= 6)
                for k = 1:size(gal_mcs, 1) # PSF component
                    accum_galaxy_pos!(
                        fs1m, bvn_derivs,
                        gal_mcs[k, j, i, s], x, wcs_jacobian,
                        is_active_source)
                end
            end
        end
    end
end

softpluslike(x) = 1000x > 1 ? 1000x - 1 : log(1000x)
softpluslikeinv(y) = y < 0 ? 1e-3exp(y) : 1e-3(y + 1)


function star_light_density!(fs0m::SensitiveFloat{T},
                             p::ImagePatch,
                             h::Int,
                             w::Int,
                             pos::Vector,
                             is_active_source::Bool) where {T <: Number}
     f(local_pos::Vector) = begin
         m_pos = Model.linear_world_to_pix(p.wcs_jacobian,
                                           p.world_center,
                                           p.pixel_center,
                                           local_pos)
         softpluslikeinv(p.itp_psf[h - m_pos[1] + 26, w - m_pos[2] + 26])
     end

     fs0m.v[] = f(pos)

     if fs0m.has_gradient && is_active_source
         ForwardDiff.gradient!(fs0m.d, f, pos)
     end

     if fs0m.has_hessian && is_active_source
         ForwardDiff.hessian!(fs0m.h, f, pos)
     end
end


"""
Add the contributions of a galaxy component term to the ELBO by
updating fs1m in place.
"""
function accum_galaxy_pos!(fs1m::SensitiveFloat{T},
                           bvn_derivs::BivariateNormalDerivatives{T},
                           gcc::GalaxyCacheComponent{T},
                           x::SVector{2,Float64},
                           wcs_jacobian::Array{Float64, 2},
                           is_active_source::Bool) where {T<:Number}
    eval_bvn_pdf!(bvn_derivs, gcc.bmc, x)
    f = bvn_derivs.f_pre[1] * gcc.gal_frac_dev_i
    fs1m.v[] += f

    if fs1m.has_gradient && is_active_source
        get_bvn_derivs!(bvn_derivs, gcc.bmc,
                        fs1m.has_gradient, fs1m.has_hessian)
        transform_bvn_derivs!(
            bvn_derivs, gcc.sig_sf, wcs_jacobian, fs1m.has_hessian)

        @aliasscope begin
            bvn_u_d = Const(bvn_derivs.bvn_u_d)
            bvn_s_d = Const(bvn_derivs.bvn_s_d)
            bvn_derivs_f_pre = Const(bvn_derivs.f_pre)

            # Accumulate the derivatives.
            @inbounds for u_id in 1:2
                fs1m.d[gal_ids.pos[u_id]] += f * bvn_u_d[u_id]
            end

            @inbounds for gal_id in 1:length(gal_shape_ids)
                fs1m.d[gal_shape_alignment[gal_id]] += f * bvn_s_d[gal_id]
            end

            # The gal_frac_dev derivative. gal_frac_dev just scales the entire component.
            # The direction is positive or negative depending on whether this
            # is an exp or dev component.
            @inbounds fs1m.d[gal_ids.gal_frac_dev] += gcc.gal_frac_dev_dir * bvn_derivs_f_pre[1]

            if fs1m.has_hessian
                # The Hessians:
                bvn_uu_h = Const(bvn_derivs.bvn_uu_h)
                bvn_ss_h = Const(bvn_derivs.bvn_ss_h)
                bvn_us_h = Const(bvn_derivs.bvn_us_h)
                gal_ids_u = Const(gal_ids.pos)

                # Hessian terms involving only the shape parameters.
                @inbounds for shape_id1 in 1:length(gal_shape_ids)
                    @inbounds for shape_id2 in 1:length(gal_shape_ids)
                        s1 = gal_shape_alignment[shape_id1]
                        s2 = gal_shape_alignment[shape_id2]
                        fs1m.h[s1, s2] +=
                            f * (bvn_ss_h[shape_id1, shape_id2] +
                                     bvn_s_d[shape_id1] * bvn_s_d[shape_id2])
                    end
                end

                # Hessian terms involving only the location parameters.
                @inbounds for u_id1 in 1:2
                    @inbounds for u_id2 in 1:2
                        u1 = gal_ids_u[u_id1]
                        u2 = gal_ids_u[u_id2]
                        fs1m.h[u1, u2] +=
                            f * (bvn_uu_h[u_id1, u_id2] + bvn_u_d[u_id1] * bvn_u_d[u_id2])
                    end
                end

                # Hessian terms involving both the shape and location parameters.
                for u_id in 1:2
                  @inbounds for shape_id in 1:length(gal_shape_ids)
                    ui = gal_ids_u[u_id]
                    si = gal_shape_alignment[shape_id]
                    fs1m.h[ui, si] +=
                        f * (bvn_us_h[u_id, shape_id] + bvn_u_d[u_id] * bvn_s_d[shape_id])
                    fs1m.h[si, ui] = fs1m.h[ui, si]
                  end
                end

                # Do the gal_frac_dev hessian terms.
                devi = gal_ids.gal_frac_dev
                @inbounds for u_id in 1:2
                    ui = gal_ids.pos[u_id]
                    fs1m.h[ui, devi] +=
                        bvn_derivs_f_pre[1] * gcc.gal_frac_dev_dir * bvn_u_d[u_id]
                    fs1m.h[devi, ui] = fs1m.h[ui, devi]
                end
                @inbounds for shape_id in 1:length(gal_shape_ids)
                    si = gal_shape_alignment[shape_id]
                    fs1m.h[si, devi] +=
                        bvn_derivs_f_pre[1] * gcc.gal_frac_dev_dir * bvn_s_d[shape_id]
                    fs1m.h[devi, si] = fs1m.h[si, devi]
                end
            end # if calculate hessian
        end
    end # if is_active_source
end


function write_star_nmgy!(world_pos::Array{Float64,1},
                          flux::Float64,
                          patch::ImagePatch,
                          pixels::Matrix{Float32};
                          write_to_patch::Bool=false)
    fs0m = SensitiveFloat{Float64}(length(StarPosParams), 1, false, false)
    H2, W2 = size(patch.active_pixel_bitmap)
    for w2 in 1:W2, h2 in 1:H2
        h = patch.bitmap_offset[1] + h2
        w = patch.bitmap_offset[2] + w2
        Model.star_light_density!(fs0m, patch, h, w, world_pos, false)
        if write_to_patch
            pixels[h2, w2] += fs0m.v[] * flux
        else
            pixels[h, w] += fs0m.v[] * flux
        end
    end
end


function write_galaxy_nmgy!(world_pos::Array{Float64,1},
                            flux::Float64,
                            gal_frac_dev::Float64,
                            gal_axis_ratio::Float64,
                            gal_angle::Float64,
                            gal_radius_px::Float64,
                            psf::Array{Model.PsfComponent,1},
                            patches::Array{ImagePatch, 2},
                            pixels::Matrix{Float32};
                            write_to_patch::Bool=false)
    bvn_derivs = Model.BivariateNormalDerivatives{Float64}()
    fs1m = SensitiveFloat{Float64}(length(GalaxyPosParams), 1, false, false)
    source_params = [[world_pos[1], world_pos[2], gal_frac_dev, gal_axis_ratio,
                     gal_angle, gal_radius_px],]
    star_mcs, gal_mcs = Model.load_bvn_mixtures(1, patches,
                          source_params, [1,], length(psf), 1,
                          calculate_gradient=false,
                          calculate_hessian=false)
    p = patches[1]
    H2, W2 = size(p.active_pixel_bitmap)
    for w2 in 1:W2, h2 in 1:H2
        # (h2, w2) index the local patch, while (h, w) index the image
        h = p.bitmap_offset[1] + h2
        w = p.bitmap_offset[2] + w2
        Model.populate_gal_fsm!(fs1m, bvn_derivs, 1, h, w, false, p.wcs_jacobian, gal_mcs)
        if write_to_patch
            pixels[h2, w2] += fs1m.v[] * flux
        else
            pixels[h, w] += fs1m.v[] * flux
        end
    end
end

