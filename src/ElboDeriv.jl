# Calculate values and partial derivatives of the variational ELBO.

module ElboDeriv

VERSION < v"0.4.0-dev" && using Docile
using CelesteTypes
import KL
import Util
import WCS
import WCSLIB

@doc """
SensitiveFloat objects for expectations involving r_s and c_s.

Args:
vs: A vector of variational parameters

Attributes:
Each matrix has one row for each color and a column for
star / galaxy.  Row 3 is the gamma distribute baseline brightness,
and all other rows are lognormal offsets.
- E_l_a: A B x Ia matrix of expectations and derivatives of
  color terms.  The rows are bands, and the columns
  are star / galaxy.
- E_ll_a: A B x Ia matrix of expectations and derivatives of
  squared color terms.  The rows are bands, and the columns
  are star / galaxy.
""" ->
immutable SourceBrightness
    E_l_a::Matrix{SensitiveFloat}  # [E[l|a=0], E[l]|a=1]]
    E_ll_a::Matrix{SensitiveFloat}   # [E[l^2|a=0], E[l^2]|a=1]]

    SourceBrightness(vs::Vector{Float64}) = begin
        r1 = vs[ids.r1]
        r2 = vs[ids.r2]
        c1 = vs[ids.c1]
        c2 = vs[ids.c2]

        # E_l_a has a row for each of the five colors and columns
        # for star / galaxy.
        E_l_a = Array(SensitiveFloat, B, Ia)

        for i = 1:Ia
            for b = 1:B
                E_l_a[b, i] = zero_sensitive_float(CanonicalParams)
            end

            # Index 3 is r_s and has a gamma expectation.
            E_l_a[3, i].v = r1[i] * r2[i]
            E_l_a[3, i].d[ids.r1[i]] = r2[i]
            E_l_a[3, i].d[ids.r2[i]] = r1[i]

            # The remaining indices involve c_s and have lognormal
            # expectations times E_c_3.
            E_c_3 = exp(c1[3, i] + .5 * c2[3, i])
            E_l_a[4, i].v = E_l_a[3, i].v * E_c_3
            E_l_a[4, i].d[ids.r1[i]] = E_l_a[3, i].d[ids.r1[i]] * E_c_3
            E_l_a[4, i].d[ids.r2[i]] = E_l_a[3, i].d[ids.r2[i]] * E_c_3
            E_l_a[4, i].d[ids.c1[3, i]] = E_l_a[4, i].v
            E_l_a[4, i].d[ids.c2[3, i]] = E_l_a[4, i].v * .5

            E_c_4 = exp(c1[4, i] + .5 * c2[4, i])
            E_l_a[5, i].v = E_l_a[4, i].v * E_c_4
            E_l_a[5, i].d[ids.r1[i]] = E_l_a[4, i].d[ids.r1[i]] * E_c_4
            E_l_a[5, i].d[ids.r2[i]] = E_l_a[4, i].d[ids.r2[i]] * E_c_4
            E_l_a[5, i].d[ids.c1[3, i]] = E_l_a[4, i].d[ids.c1[3, i]] * E_c_4
            E_l_a[5, i].d[ids.c2[3, i]] = E_l_a[4, i].d[ids.c2[3, i]] * E_c_4
            E_l_a[5, i].d[ids.c1[4, i]] = E_l_a[5, i].v
            E_l_a[5, i].d[ids.c2[4, i]] = E_l_a[5, i].v * .5

            E_c_2 = exp(-c1[2, i] + .5 * c2[2, i])
            E_l_a[2, i].v = E_l_a[3, i].v * E_c_2
            E_l_a[2, i].d[ids.r1[i]] = E_l_a[3, i].d[ids.r1[i]] * E_c_2
            E_l_a[2, i].d[ids.r2[i]] = E_l_a[3, i].d[ids.r2[i]] * E_c_2
            E_l_a[2, i].d[ids.c1[2, i]] = E_l_a[2, i].v * -1.
            E_l_a[2, i].d[ids.c2[2, i]] = E_l_a[2, i].v * .5

            E_c_1 = exp(-c1[1, i] + .5 * c2[1, i])
            E_l_a[1, i].v = E_l_a[2, i].v * E_c_1
            E_l_a[1, i].d[ids.r1[i]] = E_l_a[2, i].d[ids.r1[i]] * E_c_1
            E_l_a[1, i].d[ids.r2[i]] = E_l_a[2, i].d[ids.r2[i]] * E_c_1
            E_l_a[1, i].d[ids.c1[2, i]] = E_l_a[2, i].d[ids.c1[2, i]] * E_c_1
            E_l_a[1, i].d[ids.c2[2, i]] = E_l_a[2, i].d[ids.c2[2, i]] * E_c_1
            E_l_a[1, i].d[ids.c1[1, i]] = E_l_a[1, i].v * -1.
            E_l_a[1, i].d[ids.c2[1, i]] = E_l_a[1, i].v * .5
        end

        E_ll_a = Array(SensitiveFloat, B, Ia)
        for i = 1:Ia
            for b = 1:B
                E_ll_a[b, i] = zero_sensitive_float(CanonicalParams)
            end

            r2_sq = r2[i]^2
            E_ll_a[3, i].v = r1[i] * (1 + r1[i]) * r2_sq
            E_ll_a[3, i].d[ids.r1[i]] = (1 + 2 * r1[i]) * r2_sq
            E_ll_a[3, i].d[ids.r2[i]] = 2 * r1[i] * (1. + r1[i]) * r2[i]

            tmp3 = exp(2c1[3, i] + 2 * c2[3, i])
            E_ll_a[4, i].v = E_ll_a[3, i].v * tmp3
            E_ll_a[4, i].d[:] = E_ll_a[3, i].d * tmp3
            E_ll_a[4, i].d[ids.c1[3, i]] = E_ll_a[4, i].v * 2.
            E_ll_a[4, i].d[ids.c2[3, i]] = E_ll_a[4, i].v * 2.

            tmp4 = exp(2c1[4, i] + 2 * c2[4, i])
            E_ll_a[5, i].v = E_ll_a[4, i].v * tmp4
            E_ll_a[5, i].d[:] = E_ll_a[4, i].d * tmp4
            E_ll_a[5, i].d[ids.c1[4, i]] = E_ll_a[5, i].v * 2.
            E_ll_a[5, i].d[ids.c2[4, i]] = E_ll_a[5, i].v * 2.

            tmp2 = exp(-2c1[2, i] + 2 * c2[2, i])
            E_ll_a[2, i].v = E_ll_a[3, i].v * tmp2
            E_ll_a[2, i].d[:] = E_ll_a[3, i].d * tmp2
            E_ll_a[2, i].d[ids.c1[2, i]] = E_ll_a[2, i].v * -2.
            E_ll_a[2, i].d[ids.c2[2, i]] = E_ll_a[2, i].v * 2.

            tmp1 = exp(-2c1[1, i] + 2 * c2[1, i])
            E_ll_a[1, i].v = E_ll_a[2, i].v * tmp1
            E_ll_a[1, i].d[:] = E_ll_a[2, i].d * tmp1
            E_ll_a[1, i].d[ids.c1[1, i]] = E_ll_a[1, i].v * -2.
            E_ll_a[1, i].d[ids.c2[1, i]] = E_ll_a[1, i].v * 2.
        end

        new(E_l_a, E_ll_a)
    end
end


@doc """
Relevant parameters of a bivariate normal distribution.

Args:
  the_mean: The mean as a 2x1 column vector
  the_cov: The covaraiance as a 2x2 matrix
  weight: A scalar weight

Attributes:
   the_mean: The mean argument
   precision: The inverse of the_cov
   z: The weight times the normalizing constant.
""" ->
immutable BvnComponent
    the_mean::Vector{Float64}
    precision::Matrix{Float64}
    z::Float64

    BvnComponent(the_mean, the_cov, weight) = begin
        the_det = the_cov[1,1] * the_cov[2,2] - the_cov[1,2] * the_cov[2,1]
        c = 1 ./ (the_det^.5 * 2pi)
        new(the_mean, the_cov^-1, c * weight)
    end
end


@doc """
The convolution of a one galaxy component with one PSF component.

Args:
 - e_dev_dir: "Theta direction": this is 1 or -1, depending on whether
     increasing e_dev increases the weight of this GalaxyCacheComponent
     (1) or decreases it (-1).
 - e_dev_i: The weight given to this type of galaxy for this celestial object.
     This is either e_dev or (1 - e_dev).
 - gc: The galaxy component to be convolved
 - pc: The psf component to be convolved
 - u: The location of the celestial object as a 2x1 vector
 - e_axis: The ratio of the galaxy minor axis to major axis (0 < e_axis <= 1)
 - e_scale: The scale of the galaxy major axis

Attributes:
 - e_dev_dir: Same as input
 - e_dev_i: Same as input
 - bmc: A BvnComponent with the convolution.
 - dSigma: A 3x3 matrix containing the derivates of
     [Sigma11, Sigma12, Sigma22] (in the rows) with respect to
     [e_axis, e_angle, e_scale]
""" ->
immutable GalaxyCacheComponent
    e_dev_dir::Float64
    e_dev_i::Float64
    bmc::BvnComponent
    dSigma::Matrix{Float64}  # [Sigma11, Sigma12, Sigma22] x [e_axis, e_angle, e_scale]

    GalaxyCacheComponent(e_dev_dir::Float64, e_dev_i::Float64,
            gc::GalaxyComponent, pc::PsfComponent, u::Vector{Float64},
            e_axis::Float64, e_angle::Float64, e_scale::Float64) = begin
        XiXi = Util.get_bvn_cov(e_axis, e_angle, e_scale)
        mean_s = [pc.xiBar[1] + u[1], pc.xiBar[2] + u[2]]
        var_s = pc.tauBar + gc.nuBar * XiXi
        weight = pc.alphaBar * gc.etaBar  # excludes e_dev
        bmc = BvnComponent(mean_s, var_s, weight)

        dSigma = Array(Float64, 3, 3)
        cos_sin = cos(e_angle)sin(e_angle)
        sin_sq = sin(e_angle)^2
        cos_sq = cos(e_angle)^2
        dSigma[:, 1] = 2e_axis * e_scale^2 * [sin_sq, -cos_sin, cos_sq]
        dSigma[:, 2] = e_scale^2 * (e_axis^2 - 1) * [2cos_sin, sin_sq - cos_sq, -2cos_sin]
        dSigma[:, 3] = (2XiXi ./ e_scale)[[1, 2, 4]]
        dSigma .*= gc.nuBar

        new(e_dev_dir, e_dev_i, bmc, dSigma)
    end
end


@doc """
Convolve the current locations and galaxy shapes with the PSF.

Args:
 - psf: A vector of PSF components
 - mp: The current ModelParams

Returns:
 - star_mcs: An # of PSF components x # of sources array of BvnComponents
 - gal_mcs: An array of BvnComponents with indices
    - PSF component
    - Galaxy component
    - Galaxy type
    - Source
  - wcs: A world coordinate system object

The PSF contains three components, so you see lots of 3's below.
""" ->
function load_bvn_mixtures(psf::Vector{PsfComponent}, mp::ModelParams, wcs::WCSLIB.wcsprm)
    star_mcs = Array(BvnComponent, 3, mp.S)
    gal_mcs = Array(GalaxyCacheComponent, 3, 8, 2, mp.S)

    for s in 1:mp.S
        vs = mp.vp[s]
        m_pos = WCS.world_to_pixel(wcs, Float64[vs[ids.u[1]], vs[ids.u[2]]])

        # Convolve the star locations with the PSF.
        for k in 1:3
            pc = psf[k]
            mean_s = [pc.xiBar[1] + m_pos[1], pc.xiBar[2] + m_pos[2]]
            star_mcs[k, s] = BvnComponent(mean_s, pc.tauBar, pc.alphaBar)
        end

        # Convolve the galaxy representations with the PSF.
        for i = 1:Ia
            e_dev_dir = (i == 1) ? 1. : -1.
            e_dev_i = (i == 1) ? vs[ids.e_dev] : 1. - vs[ids.e_dev]
            m_pos = WCS.world_to_pixel(wcs, Float64[vs[ids.u[1]], vs[ids.u[2]]])

            # Galaxies of type 1 have 8 components, and type 2 have 6 components (?)
            for j in 1:[8,6][i]
                for k = 1:3
                    gal_mcs[k, j, i, s] = GalaxyCacheComponent(
                        e_dev_dir, e_dev_i, galaxy_prototypes[i][j], psf[k],
                        m_pos, vs[ids.e_axis], vs[ids.e_angle], vs[ids.e_scale])
                end
            end
        end
    end

    star_mcs, gal_mcs
end


@doc """
Return quantities related to the pdf of an offset bivariate normal.

Args:
  - bmc: A bivariate normal component
  - x: A 2x1 vector containing a mean offset to be applied to bmc
""" ->
function eval_bvn_pdf(bmc::BvnComponent, x::Vector{Float64})
    y1 = x[1] - bmc.the_mean[1]
    y2 = x[2] - bmc.the_mean[2]
    py1 = bmc.precision[1,1] * y1 + bmc.precision[1,2] * y2
    py2 = bmc.precision[2,1] * y1 + bmc.precision[2,2] * y2
    c_ytpy = -0.5 * (y1 * py1 + y2 * py2)
    f_denorm = exp(c_ytpy)
    py1, py2, bmc.z * f_denorm
end


@doc """
Add the contributions of a star's bivariate normal term to the ELBO,
by updating fs0m in place.

Args:
  - bmc: The component to be added
  - x: An offset for the component in pixel coordinates (e.g. a pixel location)
  - fs0m: A SensitiveFloat to which the value of the bvn likelihood
       and its derivatives with respect to x are added.
 - wcs: The world coordinate system object for this image.
""" ->
function accum_star_pos!(bmc::BvnComponent,
                         x::Vector{Float64},
                         fs0m::SensitiveFloat,
                         wcs_jacobian::Array{Float64, 2})
    py1, py2, f = eval_bvn_pdf(bmc, x)

    fs0m.v += f

    dfs0m_dpix = Float64[f .* py1, f .* py2]
    dfs0m_dworld = wcs_jacobian' * dfs0m_dpix
    fs0m.d[star_ids.u[1]] += dfs0m_dworld[1]
    fs0m.d[star_ids.u[2]] += dfs0m_dworld[2]
end


@doc """
Add the contributions of a galaxy component term to the ELBO by
updating fs1m in place.

Args:
  - gcc: The galaxy component to be added
  - x: An offset for the component in pixel coordinates (e.g. a pixel location)
  - fs1m: A SensitiveFloat to which the value of the likelihood
       and its derivatives with respect to x are added.
  - wcs: The world coordinate system object for this image.
""" ->
function accum_galaxy_pos!(gcc::GalaxyCacheComponent,
                           x::Vector{Float64},
                           fs1m::SensitiveFloat,
                           wcs_jacobian::Array{Float64, 2})
    py1, py2, f_pre = eval_bvn_pdf(gcc.bmc, x)
    f = f_pre * gcc.e_dev_i

    fs1m.v += f

    dfs1m_dpix = Float64[f .* py1, f .* py2]
    dfs1m_dworld = wcs_jacobian' * dfs1m_dpix
    fs1m.d[gal_ids.u[1]] += dfs1m_dworld[1]
    fs1m.d[gal_ids.u[2]] += dfs1m_dworld[2]

    fs1m.d[gal_ids.e_dev] += gcc.e_dev_dir * f_pre

    df_dSigma = (
        0.5 * f * (py1 * py1 - gcc.bmc.precision[1, 1]),
        f * (py1 * py2 - gcc.bmc.precision[1, 2]),  # NB: 2X
        0.5 * f * (py2 * py2 - gcc.bmc.precision[2, 2]))

    for j in 1:3  # [dSigma11, dSigma12, dSigma22]
        fs1m.d[gal_ids.e_axis] += df_dSigma[j] * gcc.dSigma[j, 1]
        fs1m.d[gal_ids.e_angle] += df_dSigma[j] * gcc.dSigma[j, 2]
        fs1m.d[gal_ids.e_scale] += df_dSigma[j] * gcc.dSigma[j, 3]
    end
end


@doc """
Add up the ELBO values and derivatives for a single source
in a single band.

Args:
  - sb: The source's brightness expectations and derivatives
  - star_mcs: An array of star * PSF components.  The index
      order is PSF component x source.
  - gal_mcs: An array of galaxy * PSF components.  The index order is
      PSF component x galaxy component x galaxy type x source
  - vs: The variational parameters for this source
  - child_s: The index of this source within the tile.
  - parent_s: The global index of this source.
  - m_pos: A 2x1 vector with the pixel location in pixel coordinates
  - b: The band (1 to 5)
  - fs0m: The accumulated star contributions (updated in place)
  - fs1m: The accumulated galaxy contributions (updated in place)
  - E_G: Expected celestial signal in this band (G_{nbm})
       (updated in place)
  - var_G: Variance of G (updated in place)
  - wcs: The world coordinate system object for this image.

Returns:
  - Clears and updates fs0m, fs1m with the total
    star and galaxy contributions to the ELBO from this source
    in this band.  Adds the contributions to E_G and var_G.
""" ->
function accum_pixel_source_stats!(sb::SourceBrightness,
        star_mcs::Array{BvnComponent, 2},
        gal_mcs::Array{GalaxyCacheComponent, 4},
        vs::Vector{Float64}, child_s::Int64, parent_s::Int64,
        m_pos::Vector{Float64}, b::Int64,
        fs0m::SensitiveFloat, fs1m::SensitiveFloat,
        E_G::SensitiveFloat, var_G::SensitiveFloat,
        wcs_jacobian::Array{Float64, 2})
    # Accumulate over PSF components.
    clear!(fs0m)
    for star_mc in star_mcs[:, parent_s]
        accum_star_pos!(star_mc, m_pos, fs0m, wcs_jacobian)
    end

    clear!(fs1m)
    for i = 1:2 # Galaxy types
        for j in 1:[8,6][i] # Galaxy component
            for k = 1:3 # PSF component
                accum_galaxy_pos!(gal_mcs[k, j, i, parent_s], m_pos, fs1m, wcs_jacobian)
            end
        end
    end

    # Add the contributions of this source in this band to
    # E(G) and Var(G).

    # In the structures below, 1 = star and 2 = galaxy.
    a = vs[ids.a]
    fsm = (fs0m, fs1m)
    lf = (sb.E_l_a[b, 1].v * fs0m.v, sb.E_l_a[b, 2].v * fs1m.v)
    llff = (sb.E_ll_a[b, 1].v * fs0m.v^2, sb.E_ll_a[b, 2].v * fs1m.v^2)

    E_G_s_v = a[1] * lf[1] + a[2] * lf[2]
    E_G.v += E_G_s_v

    # These formulas for the variance of G use the fact that the
    # variational distributions of each source and band are independent.
    var_G.v -= E_G_s_v^2
    var_G.v += a[1] * llff[1] + a[2] * llff[2]

    # Add the contributions of this source in this band to
    # the derivatibes of E(G) and Var(G).

    # a derivatives:
    for i in 1:Ia
        E_G.d[ids.a[i], child_s] += lf[i]
        var_G.d[ids.a[i], child_s] -= 2 * E_G_s_v * lf[i]
        var_G.d[ids.a[i], child_s] += llff[i]
    end

    # Derivatives with respect to the spatial parameters
    for i in 1:Ia # Stars and galaxies
        for p1 in 1:length(shape_standard_alignment[i])
            p0 = shape_standard_alignment[i][p1]
            a_fd = a[i] * fsm[i].d[p1]
            a_El_fd = sb.E_l_a[b, i].v * a_fd
            E_G.d[p0, child_s] += a_El_fd
            var_G.d[p0, child_s] -= 2 * E_G_s_v * a_El_fd
            var_G.d[p0, child_s] += a_fd * sb.E_ll_a[b, i].v * 2 * fsm[i].v
        end
    end

    # Derivatives with respect to the brightness parameters.
    for i in 1:Ia # Stars and galaxies
        # TODO: use p1, once using BrightnessParams type
        for p1 in 1:length(brightness_standard_alignment[i])
            p0 = brightness_standard_alignment[i][p1]
            a_f_Eld = a[i] * fsm[i].v * sb.E_l_a[b, i].d[p0]
            E_G.d[p0, child_s] += a_f_Eld
            var_G.d[p0, child_s] -= 2 * E_G_s_v * a_f_Eld
            var_G.d[p0, child_s] += a[i] * fsm[i].v^2 * sb.E_ll_a[b, i].d[p0]
        end
    end
end


@doc """
Add the contributions of the expected value of a G term to the ELBO.

Args:
  - tile_sources: A vector of source ids influencing this tile
  - x_nbm: The photon count at this pixel
  - iota: The optical sensitivity
  - E_G: The variational expected value of G
  - var_G: The variational variance of G
  - accum: A SensitiveFloat for the ELBO which is updated

Returns:
  - Adds the contributions of E_G and var_G to accum in place.
""" ->
function accum_pixel_ret!(tile_sources::Vector{Int64},
        x_nbm::Float64, iota::Float64,
        E_G::SensitiveFloat, var_G::SensitiveFloat, ret::SensitiveFloat)
    # Accumulate the values.
    # Add the lower bound to the E_q[log(F_{nbm})] term
    ret.v += x_nbm * (log(iota) + log(E_G.v) - var_G.v / (2. * E_G.v^2))

    # Subtract the E_q[F_{nbm}] term.
    ret.v -= iota * E_G.v

    # Accumulate the derivatives.
    for child_s in 1:length(tile_sources), p in 1:size(E_G.d, 1)
        parent_s = tile_sources[child_s]

        # Derivative of the log term lower bound.
        ret.d[p, parent_s] +=
            x_nbm * (E_G.d[p, child_s] / E_G.v
                     - 0.5 * (E_G.v^2 * var_G.d[p, child_s]
                              - var_G.v * 2 * E_G.v * E_G.d[p, child_s])
                        ./  E_G.v^4)

        # Derivative of the linear term.
        ret.d[p, parent_s] -= iota * E_G.d[p, child_s]
    end
end


@doc """
Return the range of image pixels in an ImageTile.
""" ->
function tile_range(tile::ImageTile, tile_width::Int64)
    h1 = 1 + (tile.hh - 1) * tile_width
    h2 = min(tile.hh * tile_width, tile.img.H)
    w1 = 1 + (tile.ww - 1) * tile_width
    w2 = min(tile.ww * tile_width, tile.img.W)
    h1:h2, w1:w2
end


@doc """
Args:
  - tile: An ImageTile (containing tile coordinates)
  - mp: Model parameters.

Returns:
  - A vector of source ids (from 1 to mp.S) that influence
    pixels in the tile.  A source influences a tile if
    there is any overlap in their squares of influence.
""" ->
function local_sources(tile::ImageTile, mp::ModelParams)
    # Corners of the tile in pixel coordinates.
    tr = mp.tile_width / 2.  # tile width
    tc = Float64[tr + (tile.hh - 1) * mp.tile_width,
                 tr + (tile.ww - 1) * mp.tile_width] # Tile center
    tc11 = tc + Float64[-tr, -tr]
    tc12 = tc + Float64[-tr, tr]
    tc22 = tc + Float64[tr, tr]
    tc21 = tc + Float64[tr, -tr]

    tile_quad = vcat(tc11', tc12', tc22', tc21')
    pc = reduce(vcat, [ mp.patches[s].center' for s=1:mp.S ])
    pr = Float64[ mp.patches[s].radius for s=1:mp.S ]
    bool_vec = WCS.sources_near_quadrilateral(pc, pr, tile_quad, tile.img.wcs)

    (collect(1:mp.S))[bool_vec]
end


@doc """
Add a tile's contribution to the ELBO likelihood term by
modifying accum in place.

Args:
  - tile: An image tile.
  - mp: The current model parameters.
  - sbs: The current source brightnesses.
  - star_mcs: All the star * PCF components.
  - gal_mcs: All the galaxy * PCF components.
  - accum: The ELBO log likelihood to be updated.
""" ->
function elbo_likelihood!(tile::ImageTile, mp::ModelParams,
        sbs::Vector{SourceBrightness},
        star_mcs::Array{BvnComponent, 2},
        gal_mcs::Array{GalaxyCacheComponent, 4},
        accum::SensitiveFloat)
    tile_sources = local_sources(tile, mp)
    h_range, w_range = tile_range(tile, mp.tile_width)

    # For speed, if there are no sources, add the noise
    # contribution directly.
    if length(tile_sources) == 0

        # NB: not using the delta-method approximation here
        if tile.img.constant_background
            nan_pixels = isnan(tile.img.pixels[h_range, w_range])
            num_pixels = length(h_range) * length(w_range) - sum(nan_pixels)
            tile_x = sum(tile.img.pixels[h_range, w_range][!nan_pixels])
            ep = tile.img.epsilon
            accum.v += tile_x * log(ep) - num_pixels * ep
        else
            for w in w_range, h in h_range
                this_pixel = tile.img.pixels[h, w]
                if !isnan(this_pixel)
                    ep = tile.img.epsilon_mat[h, w]
                    accum.v += this_pixel * log(ep) - ep                    
                end
            end
        end
        return
    end

    # fs0m and fs1m accumulate contributions from all sources
    fs0m = zero_sensitive_float(StarPosParams)
    fs1m = zero_sensitive_float(GalaxyPosParams)

    tile_S = length(tile_sources)
    E_G = zero_sensitive_float(CanonicalParams, tile_S)
    var_G = zero_sensitive_float(CanonicalParams, tile_S)

    # Iterate over pixels that are not NaN.
    for w in w_range, h in h_range
        this_pixel = tile.img.pixels[h, w]
        if !isnan(this_pixel)
            clear!(E_G)
            if tile.img.constant_background
                E_G.v = tile.img.epsilon
                iota = tile.img.iota
            else
                E_G.v = tile.img.epsilon_mat[h, w]
                iota = tile.img.iota_vec[h]
            end
            clear!(var_G)

            m_pos = Float64[h, w]
            wcs_jacobian = WCS.pixel_world_jacobian(tile.img.wcs, m_pos)
            for child_s in 1:length(tile_sources)
                parent_s = tile_sources[child_s]
                accum_pixel_source_stats!(sbs[parent_s], star_mcs, gal_mcs,
                    mp.vp[parent_s], child_s, parent_s, m_pos, tile.img.b,
                    fs0m, fs1m, E_G, var_G, wcs_jacobian)
            end

            accum_pixel_ret!(tile_sources, this_pixel, iota,
                             E_G, var_G, accum)
        end
    end
end


@doc """
Add the expected log likelihood ELBO term for an image to accum.

Args:
  - img: An image
  - mp: The current model parameters.
  - accum: A sensitive float containing the ELBO.
""" ->
function elbo_likelihood!(img::Image, mp::ModelParams, accum::SensitiveFloat)
    accum.v += -sum(lfact(img.pixels[!isnan(img.pixels)]))

    star_mcs, gal_mcs = load_bvn_mixtures(img.psf, mp, img.wcs)

    sbs = [SourceBrightness(mp.vp[s]) for s in 1:mp.S]

    WW = int(ceil(img.W / mp.tile_width))
    HH = int(ceil(img.H / mp.tile_width))
    for ww in 1:WW, hh in 1:HH
        tile = ImageTile(hh, ww, img)
        # might get a speedup from subsetting the mp here
        elbo_likelihood!(tile, mp, sbs, star_mcs, gal_mcs, accum)
    end
end


@doc """
Return the expected log likelihood for all bands in a section
of the sky.
""" ->
function elbo_likelihood(blob::Blob, mp::ModelParams)
    # Return the expected log likelihood for all bands in a section
    # of the sky.

    ret = zero_sensitive_float(CanonicalParams, mp.S)
    for img in blob
        elbo_likelihood!(img, mp, ret)
    end
    ret
end


function subtract_kl_c!(d::Int64, i::Int64, s::Int64,
                        mp::ModelParams,
                        accum::SensitiveFloat)
    vs = mp.vp[s]
    a = vs[ids.a[i]]
    k = vs[ids.k[d, i]]

    pp_kl_cid = KL.gen_diagmvn_mvn_kl(mp.pp.c[i][1][:, d],
                                      mp.pp.c[i][2][:, :, d])
    (v, (d_c1, d_c2)) = pp_kl_cid(vs[ids.c1[:, i]],
                                        vs[ids.c2[:, i]])
    accum.v -= v * a * k
    accum.d[ids.k[d, i], s] -= a * v
    accum.d[ids.c1[:, i], s] -= a * k * d_c1
    accum.d[ids.c2[:, i], s] -= a * k * d_c2
    accum.d[ids.a[i], s] -= k * v
end


function subtract_kl_k!(i::Int64, s::Int64,
                        mp::ModelParams,
                        accum::SensitiveFloat)
    vs = mp.vp[s]
    pp_kl_ki = KL.gen_categorical_kl(mp.pp.k[i])
    (v, (d_k,)) = pp_kl_ki(mp.vp[s][ids.k[:, i]])
    accum.v -= v * vs[ids.a[i]]
    accum.d[ids.k[:, i], s] -= d_k .* vs[ids.a[i]]
    accum.d[ids.a[i], s] -= v
end


function subtract_kl_r!(i::Int64, s::Int64,
                        mp::ModelParams, accum::SensitiveFloat)
    vs = mp.vp[s]
    pp_kl_r = KL.gen_gamma_kl(mp.pp.r[i][1], mp.pp.r[i][2])
    (v, (d_r1, d_r2)) = pp_kl_r(vs[ids.r1[i]], vs[ids.r2[i]])
    accum.v -= v * vs[ids.a[i]]
    accum.d[ids.r1[i], s] -= d_r1 .* vs[ids.a[i]]
    accum.d[ids.r2[i], s] -= d_r2 .* vs[ids.a[i]]
    accum.d[ids.a[i], s] -= v
end



function subtract_kl_a!(s::Int64, mp::ModelParams, accum::SensitiveFloat)
    pp_kl_a = KL.gen_categorical_kl(mp.pp.a)
    (v, (d_a,)) = pp_kl_a(mp.vp[s][ids.a])
    accum.v -= v
    accum.d[ids.a, s] -= d_a
end


@doc """
Subtract from accum the entropy and expected prior of
the variational distribution.
""" ->
function subtract_kl!(mp::ModelParams, accum::SensitiveFloat)
    for s in 1:mp.S
        subtract_kl_a!(s, mp, accum)

        for i in 1:Ia
            subtract_kl_r!(i, s, mp, accum)
            subtract_kl_k!(i, s, mp, accum)
            for d in 1:D
                subtract_kl_c!(d, i, s, mp, accum)
            end
        end
    end
end


@doc """
Calculates and returns the ELBO and its derivatives for all the bands
of an image.

Args:
  - blob: An image.
  - mp: Model parameters.
""" ->
function elbo(blob::Blob, mp::ModelParams)
    ret = elbo_likelihood(blob, mp)
    subtract_kl!(mp, ret)
    ret
end

end

