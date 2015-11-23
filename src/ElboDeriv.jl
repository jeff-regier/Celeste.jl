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
Subtract the KL divergence from the prior for r for object type i.
""" ->
function subtract_kl_r!{NumType <: Number}(
  i::Int64, s::Int64,
  mp::ModelParams{NumType},
  accum::SensitiveFloat{CanonicalParams, NumType})

    vs = mp.vp[s]
    a = vs[ids.a[i]]

    pp_kl_r = KL.gen_normal_kl(mp.pp.r_mean[i], mp.pp.r_var[i])
    (v, (d_r1, d_r2)) = pp_kl_r(vs[ids.r1[i]], vs[ids.r2[i]])

    # The old prior:
    # pp_kl_r = KL.gen_gamma_kl(mp.pp.r[1, i], mp.pp.r[2, i])
    # (v, (d_r1, d_r2)) = pp_kl_r(vs[ids.r1[i]], vs[ids.r2[i]])

    accum.v -= v * a
    accum.d[ids.r1[i], s] -= d_r1 .* a
    accum.d[ids.r2[i], s] -= d_r2 .* a
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
    E_ll_a = Array(SensitiveFloat{CanonicalParams, NumType}, B, Ia)

    for i = 1:Ia
        ids_band_3 = Int64[ids.r1[i], ids.r2[i]]
        ids_color_1 = Int64[ids.c1[1, i], ids.c2[1, i]]
        ids_color_2 = Int64[ids.c1[2, i], ids.c2[2, i]]
        ids_color_3 = Int64[ids.c1[3, i], ids.c2[3, i]]
        ids_color_4 = Int64[ids.c1[4, i], ids.c2[4, i]]

        for b = 1:B
            E_l_a[b, i] = zero_sensitive_float(CanonicalParams, NumType)
        end

        # band 3 is the reference band, relative to which the colors are
        # specified.
        # It is denoted r_s and has a lognormal expectation.
        E_l_a[3, i].v = exp(r1[i] + 0.5 * r2[i])
        E_l_a[3, i].d[ids.r1[i]] = E_l_a[3, i].v
        E_l_a[3, i].d[ids.r2[i]] = E_l_a[3, i].v * .5
        set_hess!(E_l_a[3, i], ids.r1[i], ids.r1[i], E_l_a[3, i].v)
        set_hess!(E_l_a[3, i], ids.r1[i], ids.r2[i], E_l_a[3, i].v * 0.5)
        set_hess!(E_l_a[3, i], ids.r2[i], ids.r2[i], E_l_a[3, i].v * 0.25)

        # The remaining indices involve c_s and have lognormal
        # expectations times E_c_3.

        # band 4 = band 3 * color 3.
        E_l_a[4, i].v = exp(c1[3, i] + .5 * c2[3, i])
        E_l_a[4, i].d[ids.c1[3, i]] = E_l_a[4, i].v
        E_l_a[4, i].d[ids.c2[3, i]] = E_l_a[4, i].v * .5
        set_hess!(E_l_a[4, i], ids.c1[3, i], ids.c1[3, i], E_l_a[4, i].v)
        set_hess!(E_l_a[4, i], ids.c1[3, i], ids.c2[3, i], E_l_a[4, i].v * 0.5)
        set_hess!(E_l_a[4, i], ids.c2[3, i], ids.c2[3, i], E_l_a[4, i].v * 0.25)
        multiply_sf!(E_l_a[4, i], E_l_a[3, i], ids1=ids_color_3, ids2=ids_band_3)

        # Band 5 = band 4 * color 4.
        E_l_a[5, i].v = exp(c1[4, i] + .5 * c2[4, i])
        E_l_a[5, i].d[ids.c1[4, i]] = E_l_a[5, i].v
        E_l_a[5, i].d[ids.c2[4, i]] = E_l_a[5, i].v * .5
        set_hess!(E_l_a[5, i], ids.c1[4, i], ids.c1[4, i], E_l_a[5, i].v)
        set_hess!(E_l_a[5, i], ids.c1[4, i], ids.c2[4, i], E_l_a[5, i].v * 0.5)
        set_hess!(E_l_a[5, i], ids.c2[4, i], ids.c2[4, i], E_l_a[5, i].v * 0.25)
        multiply_sf!(E_l_a[5, i], E_l_a[4, i],
                     ids1=ids_color_4, ids2=union(ids_band_3, ids_color_3))

        # Band 2 = band 3 * color 2.
        E_l_a[2, i].v = exp(-c1[2, i] + .5 * c2[2, i])
        E_l_a[2, i].d[ids.c1[2, i]] = E_l_a[2, i].v * -1.
        E_l_a[2, i].d[ids.c2[2, i]] = E_l_a[2, i].v * .5
        set_hess!(E_l_a[2, i], ids.c1[2, i], ids.c1[2, i], E_l_a[2, i].v)
        set_hess!(E_l_a[2, i], ids.c1[2, i], ids.c2[2, i], E_l_a[2, i].v * -0.5)
        set_hess!(E_l_a[2, i], ids.c2[2, i], ids.c2[2, i], E_l_a[2, i].v * 0.25)
        multiply_sf!(E_l_a[2, i], E_l_a[3, i], ids1=ids_color_2, ids2=ids_band_3)

        # Band 1 = band 2 * color 1.
        E_l_a[1, i].v = exp(-c1[1, i] + .5 * c2[1, i])
        E_l_a[1, i].d[ids.c1[1, i]] = E_l_a[1, i].v * -1.
        E_l_a[1, i].d[ids.c2[1, i]] = E_l_a[1, i].v * .5
        set_hess!(E_l_a[1, i], ids.c1[1, i], ids.c1[1, i], E_l_a[1, i].v)
        set_hess!(E_l_a[1, i], ids.c1[1, i], ids.c2[1, i], E_l_a[1, i].v * -0.5)
        set_hess!(E_l_a[1, i], ids.c2[1, i], ids.c2[1, i], E_l_a[1, i].v * 0.25)
        multiply_sf!(E_l_a[1, i], E_l_a[2, i],
                     ids1=ids_color_1, ids2=union(ids_band_3, ids_color_2))

        ################################
        # Squared terms.

        for b = 1:B
            E_ll_a[b, i] = zero_sensitive_float(CanonicalParams, NumType)
        end

        # Band 3, the reference band.
        E_ll_a[3, i].v = exp(2 * r1[i] + 2 * r2[i])
        E_ll_a[3, i].d[ids.r1[i]] = 2 * E_ll_a[3, i].v
        E_ll_a[3, i].d[ids.r2[i]] = 2 * E_ll_a[3, i].v
        for hess_ids in [(ids.r1[i], ids.r1[i]),
                         (ids.r1[i], ids.r2[i]),
                         (ids.r2[i], ids.r2[i])]
          set_hess!(E_ll_a[3, i], hess_ids..., 4.0 * E_ll_a[3, i].v)
        end

        # Band 4 = band 3 * color 3.
        E_ll_a[4, i].v = exp(2 * c1[3, i] + 2 * c2[3, i])
        E_ll_a[4, i].d[ids.c1[3, i]] = E_ll_a[4, i].v * 2.
        E_ll_a[4, i].d[ids.c2[3, i]] = E_ll_a[4, i].v * 2.
        for hess_ids in [(ids.c1[3, i], ids.c1[3, i]),
                         (ids.c1[3, i], ids.c2[3, i]),
                         (ids.c2[3, i], ids.c2[3, i])]
          set_hess!(E_ll_a[4, i], hess_ids..., E_ll_a[4, i].v * 4.0)
        end
        multiply_sf!(E_ll_a[4, i], E_ll_a[3, i],
                     ids1=ids_color_3, ids2=ids_band_3)

        # Band 5 = band 4 * color 4.
        tmp4 = exp(2 * c1[4, i] + 2 * c2[4, i])
        E_ll_a[5, i].v = exp(2 * c1[4, i] + 2 * c2[4, i])
        E_ll_a[5, i].d[ids.c1[4, i]] = E_ll_a[5, i].v * 2.
        E_ll_a[5, i].d[ids.c2[4, i]] = E_ll_a[5, i].v * 2.
        for hess_ids in [(ids.c1[4, i], ids.c1[4, i]),
                         (ids.c1[4, i], ids.c2[4, i]),
                         (ids.c2[4, i], ids.c2[4, i])]
          set_hess!(E_ll_a[5, i], hess_ids..., E_ll_a[5, i].v * 4.0)
        end
        multiply_sf!(E_ll_a[5, i], E_ll_a[4, i],
                     ids1=ids_color_4, ids2=union(ids_band_3, ids_color_3))

        # Band 2 = band 3 * color 2
        tmp2 = exp(-2 * c1[2, i] + 2 * c2[2, i])
        E_ll_a[2, i].v = exp(-2 * c1[2, i] + 2 * c2[2, i])
        E_ll_a[2, i].d[ids.c1[2, i]] = E_ll_a[2, i].v * -2.
        E_ll_a[2, i].d[ids.c2[2, i]] = E_ll_a[2, i].v * 2.
        for hess_ids in [(ids.c1[2, i], ids.c1[2, i]),
                         (ids.c2[2, i], ids.c2[2, i])]
          set_hess!(E_ll_a[2, i], hess_ids..., E_ll_a[2, i].v * 4.0)
        end
        set_hess!(E_ll_a[2, i], ids.c1[2, i], ids.c2[2, i],
                  E_ll_a[2, i].v * -4.0)
        multiply_sf!(E_ll_a[2, i], E_ll_a[3, i],
                     ids1=ids_color_2, ids2=ids_band_3)

        # Band 1 = band 2 * color 1
        E_ll_a[1, i].v = exp(-2 * c1[1, i] + 2 * c2[1, i])
        E_ll_a[1, i].d[ids.c1[1, i]] = E_ll_a[1, i].v * -2.
        E_ll_a[1, i].d[ids.c2[1, i]] = E_ll_a[1, i].v * 2.
        for hess_ids in [(ids.c1[1, i], ids.c1[1, i]),
                         (ids.c2[1, i], ids.c2[1, i])]
          set_hess!(E_ll_a[1, i], hess_ids..., E_ll_a[1, i].v * 4.0)
        end
        set_hess!(E_ll_a[1, i], ids.c1[1, i], ids.c2[1, i],
                  E_ll_a[1, i].v * -4.0)
        multiply_sf!(E_ll_a[1, i], E_ll_a[2, i],
                     ids1=ids_color_1, ids2=union(ids_band_3, ids_color_2))

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

###################################
# Bivariate normal stuff.


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
Return quantities related to the pdf of an offset bivariate normal.

Args:
  - bmc: A bivariate normal component
  - x: A 2x1 vector containing a mean offset to be applied to bmc

Returns:
  - py1: The first row of the precision times (x - the_mean)
  - py2: The second row of the precision times (x - the_mean)
  - The density of the bivariate normal times the weight.
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


##################
# Derivatives

# TODO: make this a ParameterSet and use SensitiveFloats instead.
immutable BvnDerivIndices
  sig::Vector{Int64} # Sigma_11, Sigma_12, Sigma_22 in that order.
  x::Vector{Int64} # x1, x2 in that order
  length::Int64 # The total number of indices
end

function set_bvn_deriv_indices()
  BvnDerivIndices([1, 2, 3], [4, 5], 5)
end

const bvn_ids = set_bvn_deriv_indices();

type BvnDerivs{NumType <: Number}
  # These are indexed by bvn_ids.
  v::NumType
  d::Vector{NumType}
  h::Matrix{NumType}
end


@doc """
Calculate the value, gradient, and hessian of
  -0.5 * x' sigma^-1 x - 0.5 * log|sigma|
with respect to x and sigma.
""" ->
function get_bvn_derivs{NumType <: Number}(
    bvn::BvnComponent{NumType}, x::Vector{Float64})

  py1, py2, f_pre = eval_bvn_pdf(bvn, x);

  # TODO: the value is not really neccessary except for testing,
  # I think, and if it is,  you should get the determinant from the bvn.
  v = -0.5 * (
    (x[1] - bvn.the_mean[1]) * py1 + (x[2] - bvn.the_mean[2]) * py2 -
    log(bvn.precision[1, 1] * bvn.precision[2, 2] - bvn.precision[1, 2] ^ 2))

  d = zeros(NumType, bvn_ids.length)
  d[bvn_ids.x[1]] = -py1
  d[bvn_ids.x[2]] = -py2

  # The first term is the derivative of -0.5 * x' Sigma^{-1} x
  # The second term is the derivative of -0.5 * log|Sigma|
  d[bvn_ids.sig[1]] = 0.5 * py1 * py1 - 0.5 * bvn.precision[1, 1]
  d[bvn_ids.sig[2]] = py1 * py2             - bvn.precision[1, 2]
  d[bvn_ids.sig[3]] = 0.5 * py2 * py2 - 0.5 * bvn.precision[2, 2]

  # Hessian calculation.

  # Hessian terms involving only x
  h = zeros(NumType, bvn_ids.length, bvn_ids.length)
  h[bvn_ids.x[1], bvn_ids.x[1]] = -bvn.precision[1,1]
  h[bvn_ids.x[2], bvn_ids.x[2]] = -bvn.precision[2,2]
  h[bvn_ids.x[1], bvn_ids.x[2]] = h[bvn_ids.x[2], bvn_ids.x[1]] =
    -bvn.precision[1,2]

  # Derivatives of py1 and py2 with respect to s11, s12, s22 in that order.
  # These are used for the hessian calculations.
  dpy1_ds = Array(NumType, 3)
  dpy1_ds[1] = -py1 * bvn.precision[1,1]
  dpy1_ds[2] = -py2 * bvn.precision[1,1] - py1 * bvn.precision[1,2]
  dpy1_ds[3] = -py2 * bvn.precision[1,2]

  dpy2_ds = Array(NumType, 3)
  dpy2_ds[1] = -py1 * bvn.precision[1,2]
  dpy2_ds[2] = -py1 * bvn.precision[1,1] - py2 * bvn.precision[1,2]
  dpy2_ds[3] = -py2 * bvn.precision[2,2]

  # Derivatives of Sigma^{-1} with repsect to sigma.  These are the second
  # derivatives of log|Sigma| with respect to sigma.
  dsiginv_dsig = Array(NumType, 3, 3)
  dsiginv_dsig[1, 1] = -bvn.precision[1, 1] ^ 2
  dsiginv_dsig[1, 2] = dsiginv_dsig[2, 1] =
    -2.0 * bvn.precision[1, 1] * bvn.precision[2, 1]
  dsiginv_dsig[1, 3] = dsiginv_dsig[3, 1] = -bvn.precision[1, 2] ^ 2
  dsiginv_dsig[2, 2] =
    -2.0 * (bvn.precision[1, 1] * bvn.precision[2, 2] + bvn.precision[1, 2] ^ 2)
  dsiginv_dsig[2, 3] = dsiginv_dsig[3, 2] =
    -2.0 * bvn.precision[2, 2] * bvn.precision[1, 2]
  dsiginv_dsig[3, 3] = -bvn.precision[2, 2] ^ 2

  # Hessian terms involving only sigma
  for s_ind=1:3
    index = bvn_ids.sig[s_ind]
    h[bvn_ids.sig[1], index] = h[index, bvn_ids.sig[1]] =
      py1 * dpy1_ds[s_ind] - 0.5 * dsiginv_dsig[s_ind, 1]
    h[bvn_ids.sig[2], index] = h[index, bvn_ids.sig[2]] =
      py1 * dpy2_ds[s_ind] + py2 * dpy1_ds[s_ind] - 0.5 * dsiginv_dsig[s_ind, 2]
    h[bvn_ids.sig[3], index] = h[index, bvn_ids.sig[3]] =
      py2 * dpy2_ds[s_ind] - 0.5 * dsiginv_dsig[s_ind, 3]
  end

  # Hessian terms involving both x and sigma.
  # Note that dpyA / dxB = bvn.precision[A, B]
  for x_ind=1:2
    h[bvn_ids.sig[1], bvn_ids.x[x_ind]] =
      h[bvn_ids.x[x_ind], bvn_ids.sig[1]] =
      py1 * bvn.precision[1, x_ind]
    h[bvn_ids.sig[2], bvn_ids.x[x_ind]] =
      h[bvn_ids.x[x_ind], bvn_ids.sig[2]] =
      py1 * bvn.precision[2, x_ind] + py2 * bvn.precision[1, x_ind]
    h[bvn_ids.sig[3], bvn_ids.x[x_ind]] =
      h[bvn_ids.x[x_ind], bvn_ids.sig[3]] =
      py2 * bvn.precision[2, x_ind]
  end

  BvnDerivs{NumType}(v, d, h)
end




###############################

@doc """
The derivatives of sigma with respect to the galaxy shape parameters.  In
each case, sigma is [Sigma11, Sigma12, Sigma22], and the galaxy shape
parameters are indexed by GalaxyShapeParams.
 - j: A Jacobian matrix Sigma x GalaxyShapeParams of
      d Sigma / d GalaxyShapeParams
 - t: A Sigma x GalaxyShapeParams x GalaxyShapeParams tensor of second
      derivatives d2 Sigma / d GalaxyShapeParams d GalaxyShapeParams.
""" ->
type GalaxySigmaDerivs{NumType <: Number}
  j::Matrix{NumType}
  t::Array{NumType, 3}
end

@doc """
Args:
  - e_angle: Phi in the notes
  - e_axis: Rho in the notes
  - e_scale: Lower case sigma in the notes
  - XiXi: The value of sigma.

Note that nubar is not included.
""" ->
GalaxySigmaDerivs{NumType <: Number}(
    e_angle::NumType, e_axis::NumType, e_scale::NumType,
    XiXi::Matrix{NumType}) = begin

  cos_sin = cos(e_angle)sin(e_angle)
  sin_sq = sin(e_angle)^2
  cos_sq = cos(e_angle)^2

  j = Array(NumType, 3, length(gal_shape_ids))
  j[:, gal_shape_ids.e_axis] =
    2 * e_axis * e_scale^2 * [sin_sq, -cos_sin, cos_sq]
  j[:, gal_shape_ids.e_angle] =
    e_scale^2 * (e_axis^2 - 1) * [2cos_sin, sin_sq - cos_sq, -2cos_sin]
  j[:, gal_shape_ids.e_scale] = (2XiXi ./ e_scale)[[1, 2, 4]]

  # Second derivatives.
  t = Array(NumType, 3, length(gal_shape_ids), length(gal_shape_ids))

  # Second derivatives involving e_scale
  t[:, gal_shape_ids.e_scale, gal_shape_ids.e_scale] =
    -(2 * XiXi[1, 2, 4] ./ (e_scale ^ 2))[[1, 2, 4]]
  t[:, gal_shape_ids.e_scale, gal_shape_ids.e_axis] =
    t[:, gal_shape_ids.e_axis, gal_shape_ids.e_scale] =
    (2 * j[:, gal_shape_ids.e_axis] ./ e_scale)[[1, 2, 4]]
  t[:, gal_shape_ids.e_scale, gal_shape_ids.e_angle] =
    t[:, gal_shape_ids.e_angle, gal_shape_ids.e_scale] =
    (2 * j[:, gal_shape_ids.e_angle] ./ e_scale)[[1, 2, 4]]

  # Remaining second derivatives involving e_angle
  t[:, gal_shape_ids.e_angle, gal_shape_ids.e_angle] =
    2 * e_scale^2 * (e_axis^2 - 1) *
    [cos_sq - sin_sq, 2cos_sin, sin_sq - cos_sq]
  t[:, gal_shape_ids.e_angle, gal_shape_ids.e_axis] =
    t[:, gal_shape_ids.e_axis, gal_shape_ids.e_angle] =
    2 * e_scale^2 * e_axis * [2cos_sin, sin_sq - cos_sq, -2cos_sin]

  # The second derivative involving only e_axis.
  t[:, gal_shape_ids.e_axis, gal_shape_ids.e_axis] =
    2 * e_scale^2 * [sin_sq, -cos_sin, cos_sq]

  GalaxySigmaDerivs(j, t)
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
     [e_axis, e_angle, e_scale] (in the columns)
""" ->
immutable GalaxyCacheComponent{NumType <: Number}
    e_dev_dir::Float64
    e_dev_i::NumType
    bmc::BvnComponent{NumType}
    sig_sf::GalaxySigmaDerivs{NumType}
    # [Sigma11, Sigma12, Sigma22] x [e_axis, e_angle, e_scale]

    GalaxyCacheComponent(e_dev_dir::Float64, e_dev_i::NumType,
            gc::GalaxyComponent, pc::PsfComponent, u::Vector{NumType},
            e_axis::NumType, e_angle::NumType, e_scale::NumType) = begin
        XiXi = Util.get_bvn_cov(e_axis, e_angle, e_scale)
        mean_s = NumType[pc.xiBar[1] + u[1], pc.xiBar[2] + u[2]]
        var_s = pc.tauBar + gc.nuBar * XiXi
        weight = pc.alphaBar * gc.etaBar  # excludes e_dev
        bmc = BvnComponent(mean_s, var_s, weight)

        sig_sf = GalaxySigmaDerivs(e_angle, e_axis, e_scale, XiXi)
        sig_sf.j .*= gc.nuBar

        new(e_dev_dir, e_dev_i, bmc, sig_sf)
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
        for i = 1:2 # i indexes dev vs exp galaxy types.
            e_dev_dir = (i == 1) ? 1. : -1.
            e_dev_i = (i == 1) ? vs[ids.e_dev] : 1. - vs[ids.e_dev]

            # Galaxies of type 1 have 8 components, and type 2 have
            # 6 components.
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
Add the contributions of a star's bivariate normal term to the ELBO,
by updating fs0m in place.

Args:
  - bmc: The component to be added
  - x: An offset for the component in pixel coordinates (e.g. a pixel location)
  - fs0m: A SensitiveFloat to which the value of the bvn likelihood
       and its derivatives with respect to x are added.
 - wcs_jacobian: The jacobian of the function pixel = F(world) at this location.
""" ->
function accum_star_pos!{NumType <: Number}(
                         bmc::BvnComponent{NumType},
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

    # Note that dpyA / dxB = bmc.precision[A, B]
    # TODO: there's a redundant step here.
    # TODO: test this!
    for u_id = 1:2
      fs0m.hs[1][star_ids.u[1], star_ids.u[u_id]] +=
        convert(NumType,
                f * (wcs_jacobian[1, 1] * bmc.precision[1, u_id] +
                     wcs_jacobian[2, 1] * bmc.precision[2, u_id]))
      fs0m.hs[1][star_ids.u[2], star_ids.u[u_id]] +=
        convert(NumType,
                f * (wcs_jacobian[1, 2] * bmc.precision[1, u_id] +
                     wcs_jacobian[2, 2] * bmc.precision[2, u_id]))
    end
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
function accum_galaxy_pos!{NumType <: Number}(
                           gcc::GalaxyCacheComponent{NumType},
                           x::Vector{Float64},
                           fs1m::SensitiveFloat{GalaxyPosParams, NumType},
                           wcs_jacobian::Array{Float64, 2})
    py1, py2, f_pre = eval_bvn_pdf(gcc.bmc, x)
    f = f_pre * gcc.e_dev_i

    fs1m.v += f

    bvn_sf = get_bvn_derivs(gcc.bmc, x);

    # Gradient calculations.
    # Note that dxA_duB = -wcs_jacobian[A, B].  (It is minus the jacobian
    # because the object position affects the bvn.the_mean term.)
    for x_id in 1:2, u_id in 1:2
      fs1m.d[gal_ids.u[u_id]] +=
        -f * bvn_sf.d[bvn_ids.x[x_id]] *
        wcs_jacobian[x_id, u_id]
    end

    # The e_dev derivative.  e_dev just scales the entire component.
    fs1m.d[gal_ids.e_dev] += gcc.e_dev_dir * f_pre

    # Use the chain rule.
    gal_d = zeros(NumType, length(gal_shape_ids))
    for gal_id in 1:length(gal_shape_ids), sig_id in 1:3
      gal_d[gal_id] +=
        f * gcc.sig_sf.j[sig_id, gal_id] * bvn_sf.d[bvn_ids.sig[sig_id]]
    end

    for gal_id in 1:length(gal_shape_ids)
      fs1m.d[gal_shape_alignment[gal_id]] += gal_d[gal_id]
    end

    # Calculate the hessian.
    for gal_id1 in 1:length(gal_shape_ids), gal_id2 in 1:length(gal_shape_ids)
      g1 = gal_shape_alignment[gal_id1]
      g2 = gal_shape_alignment[gal_id2]
      fs1m.hs[1][g1, g2] += f * gal_d[gal_id1] * gal_d[gal_id1]

      for sig_id in 1:3
        fs1m.hs[1][g1, g2] += f * gcc.sig_sf.t[sig_id, gal_id1, gal_id2]
      end
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
      accum_pixel_source_stats!(
          sbs[tile_sources[child_s]], star_mcs, gal_mcs,
          mp.vp[tile_sources[child_s]], child_s, tile_sources[child_s],
          Float64[tile.h_range[h], tile.w_range[w]], tile.b,
          fs0m, fs1m, E_G, var_G,
          mp.patches[child_s, tile.b].wcs_jacobian)
  end

  # Return the appropriate value of iota.
  tile.constant_background ? tile.iota : tile.iota_vec[h]
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
