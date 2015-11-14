# Calculate values and partial derivatives of the variational ELBO.

module ElboDeriv

VERSION < v"0.4.0-dev" && using Docile
using CelesteTypes
import KL
import Util
import Polygons
import SloanDigitalSkySurvey: WCS
import WCSLIB

using DualNumbers.Dual

export tile_predicted_image
export ParameterMessage, update_parameter_message!

@doc """
Subtract the KL divergence from the prior for c
""" ->
function subtract_kl_c!{NumType <: Number}(
  d::Int64, i::Int64, s::Int64,
  mp::ModelParams{NumType},
  accum::SensitiveFloat{CanonicalParams, NumType})

    vs = mp.vp[s]
    a = vs[ids.a[i]]
    k = vs[ids.k[d, i]]

    pp_kl_cid = KL.gen_diagmvn_mvn_kl(mp.pp.c_mean[:, d, i],
                                      mp.pp.c_cov[:, :, d, i])
    (v, (d_c1, d_c2)) = pp_kl_cid(vs[ids.c1[:, i]],
                                        vs[ids.c2[:, i]])
    accum.v -= v * a * k
    accum.d[ids.k[d, i], s] -= a * v
    accum.d[ids.c1[:, i], s] -= a * k * d_c1
    accum.d[ids.c2[:, i], s] -= a * k * d_c2
    accum.d[ids.a[i], s] -= k * v
end

@doc """
Subtract the KL divergence from the prior for k
""" ->
function subtract_kl_k!{NumType <: Number}(
  i::Int64, s::Int64,
  mp::ModelParams{NumType},
  accum::SensitiveFloat{CanonicalParams, NumType})

    vs = mp.vp[s]
    pp_kl_ki = KL.gen_categorical_kl(mp.pp.k[:, i])
    (v, (d_k,)) = pp_kl_ki(mp.vp[s][ids.k[:, i]])
    accum.v -= v * vs[ids.a[i]]
    accum.d[ids.k[:, i], s] -= d_k .* vs[ids.a[i]]
    accum.d[ids.a[i], s] -= v
end


@doc """
Subtract the KL divergence from the prior for r
""" ->
function subtract_kl_r!{NumType <: Number}(
  i::Int64, s::Int64,
  mp::ModelParams{NumType},
  accum::SensitiveFloat{CanonicalParams, NumType})
    vs = mp.vp[s]
    pp_kl_r = KL.gen_gamma_kl(mp.pp.r[1, i], mp.pp.r[2, i])
    (v, (d_r1, d_r2)) = pp_kl_r(vs[ids.r1[i]], vs[ids.r2[i]])
    accum.v -= v * vs[ids.a[i]]
    accum.d[ids.r1[i], s] -= d_r1 .* vs[ids.a[i]]
    accum.d[ids.r2[i], s] -= d_r2 .* vs[ids.a[i]]
    accum.d[ids.a[i], s] -= v
end


@doc """
Subtract the KL divergence from the prior for a
""" ->
function subtract_kl_a!{NumType <: Number}(
  s::Int64, mp::ModelParams{NumType},
  accum::SensitiveFloat{CanonicalParams, NumType})
    pp_kl_a = KL.gen_categorical_kl(mp.pp.a)
    (v, (d_a,)) = pp_kl_a(mp.vp[s][ids.a])
    accum.v -= v
    accum.d[ids.a, s] -= d_a
end


@doc """
Subtract from accum the entropy and expected prior of
the variational distribution.
""" ->
function subtract_kl!{NumType <: Number}(
  mp::ModelParams{NumType}, accum::SensitiveFloat{CanonicalParams, NumType})
    for s in mp.active_sources
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
immutable SourceBrightness{NumType <: Number}
    # [E[l|a=0], E[l]|a=1]]
    E_l_a::Matrix{SensitiveFloat{CanonicalParams, NumType}}

    # [E[l^2|a=0], E[l^2]|a=1]]
    E_ll_a::Matrix{SensitiveFloat{CanonicalParams, NumType}}
end


SourceBrightness{NumType <: Number}(vs::Vector{NumType}) = begin
    r1 = vs[ids.r1]
    r2 = vs[ids.r2]
    c1 = vs[ids.c1]
    c2 = vs[ids.c2]

    # E_l_a has a row for each of the five colors and columns
    # for star / galaxy.
    E_l_a = Array(SensitiveFloat{CanonicalParams, NumType}, B, Ia)

    for i = 1:Ia
        for b = 1:B
            E_l_a[b, i] = zero_sensitive_float(CanonicalParams, NumType)
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

    E_ll_a = Array(SensitiveFloat{CanonicalParams, NumType}, B, Ia)
    for i = 1:Ia
        for b = 1:B
            E_ll_a[b, i] = zero_sensitive_float(CanonicalParams, NumType)
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

    SourceBrightness(E_l_a, E_ll_a)
end


@doc """
A convenience function for getting only the brightness parameters
from model parameters.

Args:
  mp: Model parameters

Returns:
  An array of E_l_a and E_ll_a for each source.
""" ->
function get_brightness{NumType <: Number}(mp::ModelParams{NumType})
    brightness = [SourceBrightness(mp.vp[s]) for s in mp.S];
    brightness_vals = [ Float64[b.E_l_a[i, j].v for
        i=1:size(b.E_l_a, 1), j=1:size(b.E_l_a, 2)] for b in brightness]
    brightness_squares = [ Float64[b.E_l_a[i, j].v for
        i=1:size(b.E_ll_a, 1), j=1:size(b.E_ll_a, 2)] for b in brightness]

    brightness_vals, brightness_squares
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
immutable BvnComponent{NumType <: Number}
    the_mean::Vector{NumType}
    precision::Matrix{NumType}
    z::NumType

    BvnComponent(the_mean, the_cov, weight) = begin
        the_det = the_cov[1,1] * the_cov[2,2] - the_cov[1,2] * the_cov[2,1]
        c = 1 ./ (the_det^.5 * 2pi)
        new(the_mean, the_cov^-1, c * weight)
    end
end

BvnComponent{NumType <: Number}(
  the_mean::Vector{NumType}, the_cov::Matrix{Float64}, weight::Float64) = begin
    BvnComponent{NumType}(
      the_mean, convert(Array{NumType}, the_cov), convert(NumType, weight))
end

BvnComponent{NumType <: Number}(
  the_mean::Vector{NumType}, the_cov::Matrix{NumType}, weight::Float64) = begin
    BvnComponent{NumType}(the_mean, the_cov, convert(NumType, weight))
end

BvnComponent{NumType <: Number}(
  the_mean::Vector{NumType}, the_cov::Matrix{NumType}, weight::NumType) = begin
    BvnComponent{NumType}(the_mean, the_cov, weight)
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
immutable GalaxyCacheComponent{NumType <: Number}
    e_dev_dir::Float64
    e_dev_i::NumType
    bmc::BvnComponent{NumType}
    dSigma::Matrix{NumType}
    # [Sigma11, Sigma12, Sigma22] x [e_axis, e_angle, e_scale]

    GalaxyCacheComponent(e_dev_dir::Float64, e_dev_i::NumType,
            gc::GalaxyComponent, pc::PsfComponent, u::Vector{NumType},
            e_axis::NumType, e_angle::NumType, e_scale::NumType) = begin
        XiXi = Util.get_bvn_cov(e_axis, e_angle, e_scale)
        mean_s = NumType[pc.xiBar[1] + u[1], pc.xiBar[2] + u[2]]
        var_s = pc.tauBar + gc.nuBar * XiXi
        weight = pc.alphaBar * gc.etaBar  # excludes e_dev
        bmc = BvnComponent(mean_s, var_s, weight)

        dSigma = Array(NumType, 3, 3)
        cos_sin = cos(e_angle)sin(e_angle)
        sin_sq = sin(e_angle)^2
        cos_sq = cos(e_angle)^2
        dSigma[:, 1] = 2e_axis * e_scale^2 * [sin_sq, -cos_sin, cos_sq]
        dSigma[:, 2] = e_scale^2 * (e_axis^2 - 1) *
                       [2cos_sin, sin_sq - cos_sq, -2cos_sin]
        dSigma[:, 3] = (2XiXi ./ e_scale)[[1, 2, 4]]
        dSigma .*= gc.nuBar

        new(e_dev_dir, e_dev_i, bmc, dSigma)
    end
end

GalaxyCacheComponent{NumType <: Number}(e_dev_dir::Float64, e_dev_i::NumType,
                     gc::GalaxyComponent, pc::PsfComponent, u::Vector{NumType},
                     e_axis::NumType, e_angle::NumType, e_scale::NumType) =
    GalaxyCacheComponent{NumType}(
      e_dev_dir, e_dev_i, gc, pc, u, e_axis, e_angle, e_scale)


@doc """
Convolve the current locations and galaxy shapes with the PSF.

Args:
 - psf: A vector of PSF components
 - mp: The current ModelParams
 - b: The current band

Returns:
 - star_mcs: An # of PSF components x # of sources array of BvnComponents
 - gal_mcs: An array of BvnComponents with indices
    - PSF component
    - Galaxy component
    - Galaxy type
    - Source

The PSF contains three components, so you see lots of 3's below.
""" ->
function load_bvn_mixtures{NumType <: Number}(mp::ModelParams{NumType}, b::Int64)
    star_mcs = Array(BvnComponent{NumType}, 3, mp.S)
    gal_mcs = Array(GalaxyCacheComponent{NumType}, 3, 8, 2, mp.S)

    for s in 1:mp.S
        psf = mp.patches[s, b].psf
        vs = mp.vp[s]

        world_loc = vs[[ids.u[1], ids.u[2]]]
        m_pos = WCS.world_to_pixel(mp.patches[s, b].wcs_jacobian,
                                   mp.patches[s, b].center,
                                   mp.patches[s, b].pixel_center, world_loc)

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

            # Galaxies of type 1 have 8 components, and type 2 have
            # 6 components (?)
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
function eval_bvn_pdf{NumType <: Number}(
  bmc::BvnComponent{NumType}, x::Vector{Float64})

    y1 = x[1] - bmc.the_mean[1]
    y2 = x[2] - bmc.the_mean[2]
    py1 = bmc.precision[1,1] * y1 + bmc.precision[1,2] * y2
    py2 = bmc.precision[2,1] * y1 + bmc.precision[2,2] * y2
    c_ytpy = -0.5 * (y1 * py1 + y2 * py2)
    f_denorm = exp(c_ytpy)
    py1, py2, bmc.z * f_denorm
end


@doc """
A type containing all the information that needs to be communicated
to worker nodes at each iteration.  This currently consists of pre-computed
information about each source.

Attributes:
  vp: The VariationalParams for the ModelParams object
  star_mcs_vec: A vector of star BVN components, one for each band
  gal_mcs_vec: A vector of galaxy BVN components, one for each band
  sbs_vec: A vector of brightness vectors, one for each band
""" ->
type ParameterMessage{NumType <: Number}
  vp::VariationalParams{NumType}
  star_mcs_vec::Vector{Array{BvnComponent{NumType},2}}
  gal_mcs_vec::Vector{Array{GalaxyCacheComponent{NumType},4}}
  sbs_vec::Vector{Vector{SourceBrightness{NumType}}}
end

@doc """
This allocates memory for but does not initialize the source parameters.
""" ->
ParameterMessage{NumType <: Number}(mp::ModelParams{NumType}) = begin
  num_bands = size(mp.patches)[2]
  star_mcs_vec = Array(Array{BvnComponent{NumType},2}, num_bands)
  gal_mcs_vec = Array(Array{GalaxyCacheComponent{NumType},4}, num_bands)
  sbs_vec = Array(Vector{SourceBrightness{NumType}}, num_bands)
  ParameterMessage(mp.vp, star_mcs_vec, gal_mcs_vec, sbs_vec)
end


@doc """
Update a ParameterMessage in place using mp.

Args:
  - mp: A ModelParams object
  - param_msg: A ParameterMessage that is updated using the parameter values
               in mp.
""" ->
function update_parameter_message!{NumType <: Number}(
    mp::ModelParams{NumType}, param_msg::ParameterMessage{NumType})
  for b=1:5
    param_msg.star_mcs_vec[b], param_msg.gal_mcs_vec[b] =
      load_bvn_mixtures(mp, b);
    param_msg.sbs_vec[b] = SourceBrightness{NumType}[
      SourceBrightness(mp.vp[s]) for s in 1:mp.S];
  end
end


@doc """
Add the contributions of a star's bivariate normal term to the ELBO,
by updating fs0m in place.

Args:
  - bmc: The component to be added
  - x: An offset for the component in pixel coordinates (e.g. a pixel location)
  - fs0m: A SensitiveFloat to which the value of the bvn likelihood
       and its derivatives with respect to x are added.
 - wcs_jacobian: The jacobian of the function pixel = F(world) at this location.
""" ->
function accum_star_pos!{NumType <: Number}(bmc::BvnComponent{NumType},
                         x::Vector{Float64},
                         fs0m::SensitiveFloat{StarPosParams, NumType},
                         wcs_jacobian::Array{Float64, 2})
    py1, py2, f = eval_bvn_pdf(bmc, x)

    fs0m.v += f

    # This is
    # dfs0m_dworld = wcs_jacobian' * NumType[f .* py1, f .* py2]
    fs0m.d[star_ids.u[1]] +=
      convert(NumType,
              f * (wcs_jacobian[1, 1] * py1 + wcs_jacobian[2, 1] * py2))
    fs0m.d[star_ids.u[2]] +=
      convert(NumType,
              f * (wcs_jacobian[1, 2] * py1 + wcs_jacobian[2, 2] * py2))
end


@doc """
Add the contributions of a galaxy component term to the ELBO by
updating fs1m in place.

Args:
  - gcc: The galaxy component to be added
  - x: An offset for the component in pixel coordinates (e.g. a pixel location)
  - fs1m: A SensitiveFloat to which the value of the likelihood
       and its derivatives with respect to x are added.
 - wcs_jacobian: The jacobian of the function pixel = F(world) at this location.
""" ->
function accum_galaxy_pos!{NumType <: Number}(gcc::GalaxyCacheComponent{NumType},
                           x::Vector{Float64},
                           fs1m::SensitiveFloat{GalaxyPosParams, NumType},
                           wcs_jacobian::Array{Float64, 2})
    py1, py2, f_pre = eval_bvn_pdf(gcc.bmc, x)
    f = f_pre * gcc.e_dev_i

    fs1m.v += f

    # This is
    # dfs1m_dworld = wcs_jacobian' * NumType[f .* py1, f .* py2]
    fs1m.d[gal_ids.u[1]] +=
      convert(NumType,
              f * (wcs_jacobian[1, 1] * py1 + wcs_jacobian[2, 1] * py2))
    fs1m.d[gal_ids.u[2]] +=
      convert(NumType,
              f * (wcs_jacobian[1, 2] * py1 + wcs_jacobian[2, 2] * py2))

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
  - wcs_jacobian: The jacobian of the function pixel = F(world) at this location.

Returns:
  - Clears and updates fs0m, fs1m with the total
    star and galaxy contributions to the ELBO from this source
    in this band.  Adds the contributions to E_G and var_G.
""" ->
function accum_pixel_source_stats!{NumType <: Number}(
        sb::SourceBrightness{NumType},
        star_mcs::Array{BvnComponent{NumType}, 2},
        gal_mcs::Array{GalaxyCacheComponent{NumType}, 4},
        vs::Vector{NumType}, child_s::Int64, parent_s::Int64,
        m_pos::Vector{Float64}, b::Int64,
        fs0m::SensitiveFloat{StarPosParams, NumType},
        fs1m::SensitiveFloat{GalaxyPosParams, NumType},
        E_G::SensitiveFloat{CanonicalParams, NumType},
        var_G::SensitiveFloat{CanonicalParams, NumType},
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
                accum_galaxy_pos!(
                  gal_mcs[k, j, i, parent_s], m_pos, fs1m, wcs_jacobian)
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
    # the derivatives of E(G) and Var(G).

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
function accum_pixel_ret!{NumType <: Number}(
        tile_sources::Vector{Int64},
        x_nbm::Float64, iota::Float64,
        E_G::SensitiveFloat{CanonicalParams, NumType},
        var_G::SensitiveFloat{CanonicalParams, NumType},
        ret::SensitiveFloat{CanonicalParams, NumType})
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
Expected pixel brightness.
Args:
  h: The row of the tile
  w: The column of the tile
  ...the rest are the same as elsewhere.

Returns:
  - Iota.
""" ->
function expected_pixel_brightness!{NumType <: Number}(
    h::Int64, w::Int64,
    sbs::Vector{SourceBrightness{NumType}},
    star_mcs::Array{BvnComponent{NumType}, 2},
    gal_mcs::Array{GalaxyCacheComponent{NumType}, 4},
    tile::ImageTile,
    E_G::SensitiveFloat{CanonicalParams, NumType},
    var_G::SensitiveFloat{CanonicalParams, NumType},
    mp::ModelParams{NumType},
    tile_sources::Vector{Int64},
    fs0m::SensitiveFloat{StarPosParams, NumType},
    fs1m::SensitiveFloat{GalaxyPosParams, NumType};
    include_epsilon::Bool=true)

  clear!(E_G)
  clear!(var_G)

  if include_epsilon
    E_G.v = tile.constant_background ? tile.epsilon : tile.epsilon_mat[h, w]
  else
    E_G.v = 0.0
  end

  for child_s in 1:length(tile_sources)
      accum_pixel_source_stats!(sbs[tile_sources[child_s]], star_mcs, gal_mcs,
          mp.vp[tile_sources[child_s]], child_s, tile_sources[child_s],
          Float64[tile.h_range[h], tile.w_range[w]], tile.b,
          fs0m, fs1m, E_G, var_G,
          mp.patches[child_s, tile.b].wcs_jacobian)
  end

  # Return the appropriate value of iota.
  tile.constant_background ? tile.iota : tile.iota_vec[h]
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
function tile_likelihood!{NumType <: Number}(
        tile::ImageTile,
        tile_sources::Vector{Int64},
        mp::ModelParams{NumType},
        sbs::Vector{SourceBrightness{NumType}},
        star_mcs::Array{BvnComponent{NumType}, 2},
        gal_mcs::Array{GalaxyCacheComponent{NumType}, 4},
        accum::SensitiveFloat{CanonicalParams, NumType};
        include_epsilon::Bool=true)

    # For speed, if there are no sources, add the noise
    # contribution directly.
    if (length(tile_sources) == 0) && include_epsilon
        # NB: not using the delta-method approximation here
        if tile.constant_background
            nan_pixels = Base.isnan(tile.pixels)
            num_pixels =
              length(tile.h_range) * length(tile.w_range) - sum(nan_pixels)
            tile_x = sum(tile.pixels[!nan_pixels])
            ep = tile.epsilon
            accum.v += tile_x * log(ep) - num_pixels * ep
        else
            for w in 1:tile.w_width, h in 1:tile.h_width
                this_pixel = tile.pixels[h, w]
                if !Base.isnan(this_pixel)
                    ep = tile.epsilon_mat[h, w]
                    accum.v += this_pixel * log(ep) - ep
                end
            end
        end
        return
    end

    # fs0m and fs1m accumulate contributions from all sources.
    fs0m = zero_sensitive_float(StarPosParams, NumType)
    fs1m = zero_sensitive_float(GalaxyPosParams, NumType)

    tile_S = length(tile_sources)
    E_G = zero_sensitive_float(CanonicalParams, NumType, tile_S)
    var_G = zero_sensitive_float(CanonicalParams, NumType, tile_S)

    # Iterate over pixels that are not NaN.
    for w in 1:tile.w_width, h in 1:tile.h_width
        this_pixel = tile.pixels[h, w]
        if !Base.isnan(this_pixel)
            iota = expected_pixel_brightness!(
              h, w, sbs, star_mcs, gal_mcs, tile, E_G, var_G,
              mp, tile_sources, fs0m, fs1m,
              include_epsilon=include_epsilon)
            accum_pixel_ret!(tile_sources, this_pixel, iota, E_G, var_G, accum)
        end
    end

    # Subtract the log factorial term
    accum.v += -sum(lfact(tile.pixels[!Base.isnan(tile.pixels)]))
end


@doc """
Return the image predicted for the tile given the current parameters.

Args:
  - tile: An image tile.
  - mp: The current model parameters.
  - sbs: The current source brightnesses.
  - star_mcs: All the star * PCF components.
  - gal_mcs: All the galaxy * PCF components.
  - accum: The ELBO log likelihood to be updated.

Returns:
  A matrix of the same size as the tile with the predicted brightnesses.
""" ->
function tile_predicted_image{NumType <: Number}(
        tile::ImageTile,
        tile_sources::Vector{Int64},
        mp::ModelParams{NumType},
        sbs::Vector{SourceBrightness{NumType}},
        star_mcs::Array{BvnComponent{NumType}, 2},
        gal_mcs::Array{GalaxyCacheComponent{NumType}, 4},
        accum::SensitiveFloat{CanonicalParams, NumType};
        include_epsilon::Bool=true)

    # fs0m and fs1m accumulate contributions from all sources.
    fs0m = zero_sensitive_float(StarPosParams, NumType)
    fs1m = zero_sensitive_float(GalaxyPosParams, NumType)

    tile_S = length(tile_sources)
    E_G = zero_sensitive_float(CanonicalParams, NumType, tile_S)
    var_G = zero_sensitive_float(CanonicalParams, NumType, tile_S)

    predicted_pixels = copy(tile.pixels)
    # Iterate over pixels that are not NaN.
    for w in 1:tile.w_width, h in 1:tile.h_width
        this_pixel = tile.pixels[h, w]
        if !Base.isnan(this_pixel)
            iota = expected_pixel_brightness!(
              h, w, sbs, star_mcs, gal_mcs, tile, E_G, var_G,
              mp, tile_sources, fs0m, fs1m,
              include_epsilon=include_epsilon)
            predicted_pixels[h, w] = E_G.v * iota
        end
    end

    predicted_pixels
end


@doc """
Produce a predicted image for a given tile and model parameters.

If include_epsilon is true, then the background is also rendered.
Otherwise, only pixels from the object are rendered.
""" ->
function tile_predicted_image{NumType <: Number}(
    tile::ImageTile, mp::ModelParams{NumType};
    include_epsilon::Bool=false)

  b = tile.b
  star_mcs, gal_mcs = load_bvn_mixtures(mp, b)
  sbs = [SourceBrightness(mp.vp[s]) for s in 1:mp.S]

  accum = zero_sensitive_float(CanonicalParams, NumType, mp.S)
  tile_sources = mp.tile_sources[b][tile.hh, tile.ww]

  tile_predicted_image(tile,
                       tile_sources,
                       mp,
                       sbs,
                       star_mcs,
                       gal_mcs,
                       accum,
                       include_epsilon=include_epsilon)
end


@doc """
The ELBO likelihood for given brighntess and bvn components.
""" ->
function elbo_likelihood!{NumType <: Number}(
  tiled_image::Array{ImageTile},
  mp::ModelParams{NumType},
  sbs::Vector{SourceBrightness{NumType}},
  star_mcs::Array{BvnComponent{NumType}, 2},
  gal_mcs::Array{GalaxyCacheComponent{NumType}, 4},
  accum::SensitiveFloat{CanonicalParams, NumType})

  @assert maximum(mp.active_sources) <= mp.S
  for tile in tiled_image[:]
    tile_sources = mp.tile_sources[tile.b][tile.hh, tile.ww]
    if length(intersect(tile_sources, mp.active_sources)) > 0
      tile_likelihood!(
        tile, tile_sources, mp, sbs, star_mcs, gal_mcs, accum);
    end
  end

end


@doc """
Evaluate the ELBO with pre-computed brightnesses and components
stored in ParameterMessage.
""" ->
function elbo_likelihood!{NumType <: Number}(
    tiled_blob::TiledBlob,
    param_msg::ParameterMessage{NumType},
    mp::ModelParams{NumType},
    accum::SensitiveFloat{CanonicalParams, NumType})

  clear!(accum)
  mp.vp = param_msg.vp
  for b in 1:5
    sbs = param_msg.sbs_vec[b]
    star_mcs = param_msg.star_mcs_vec[b]
    gal_mcs = param_msg.gal_mcs_vec[b]
    elbo_likelihood!(tiled_blob[b], mp, sbs, star_mcs, gal_mcs, accum)
  end
end


@doc """
Add the expected log likelihood ELBO term for an image to accum.

Args:
  - tiles: An array of ImageTiles
  - mp: The current model parameters.
  - accum: A sensitive float containing the ELBO.
  - b: The current band
""" ->
function elbo_likelihood!{NumType <: Number}(
  tiles::Array{ImageTile}, mp::ModelParams{NumType},
  b::Int64, accum::SensitiveFloat{CanonicalParams, NumType})

  star_mcs, gal_mcs = load_bvn_mixtures(mp, b)
  sbs = SourceBrightness{NumType}[SourceBrightness(mp.vp[s]) for s in 1:mp.S]
  elbo_likelihood!(tiles, mp, sbs, star_mcs, gal_mcs, accum)
end


@doc """
Return the expected log likelihood for all bands in a section
of the sky.
""" ->
function elbo_likelihood{NumType <: Number}(
  tiled_blob::TiledBlob, mp::ModelParams{NumType})
    # Return the expected log likelihood for all bands in a section
    # of the sky.

    ret = zero_sensitive_float(CanonicalParams, NumType, mp.S)
    for b in 1:length(tiled_blob)
        elbo_likelihood!(tiled_blob[b], mp, b, ret)
    end
    ret
end


@doc """
Calculates and returns the ELBO and its derivatives for all the bands
of an image.

Args:
  - tiled_blob: A TiledBlob.
  - mp: Model parameters.
""" ->
function elbo{NumType <: Number}(tiled_blob::TiledBlob, mp::ModelParams{NumType})
    ret = elbo_likelihood(tiled_blob, mp)
    subtract_kl!(mp, ret)
    ret
end



end
