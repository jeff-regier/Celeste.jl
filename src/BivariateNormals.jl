
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

# TODO: make this a ParameterSet and use SensitiveFloats instead?
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
  # I think, and if it is, you should get the determinant from the bvn.
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
    (2 * XiXi ./ (e_scale ^ 2))[[1, 2, 4]]
  t[:, gal_shape_ids.e_scale, gal_shape_ids.e_axis] =
    (2 * j[:, gal_shape_ids.e_axis] ./ e_scale)
  t[:, gal_shape_ids.e_scale, gal_shape_ids.e_angle] =
    (2 * j[:, gal_shape_ids.e_angle] ./ e_scale)

  t[:, gal_shape_ids.e_axis, gal_shape_ids.e_scale] =
    t[:, gal_shape_ids.e_scale, gal_shape_ids.e_axis]
  t[:, gal_shape_ids.e_angle, gal_shape_ids.e_scale] =
    t[:, gal_shape_ids.e_scale, gal_shape_ids.e_angle]

  # Remaining second derivatives involving e_angle
  t[:, gal_shape_ids.e_angle, gal_shape_ids.e_angle] =
    2 * e_scale^2 * (e_axis^2 - 1) *
    [cos_sq - sin_sq, 2cos_sin, sin_sq - cos_sq]
  t[:, gal_shape_ids.e_angle, gal_shape_ids.e_axis] =
    2 * e_scale^2 * e_axis * [2cos_sin, sin_sq - cos_sq, -2cos_sin]
  t[:, gal_shape_ids.e_axis, gal_shape_ids.e_angle] =
    t[:, gal_shape_ids.e_angle, gal_shape_ids.e_axis]

  # The second derivative involving only e_axis.
  t[:, gal_shape_ids.e_axis, gal_shape_ids.e_axis] =
    2 * e_scale^2 * [sin_sq, -cos_sin, cos_sq]

  GalaxySigmaDerivs(j, t)
end


@doc """
The convolution of a one galaxy component with one PSF component.
It also contains the derivatives of sigma with respect to the shape parameters.
It does not contain the derivatives with respect to other parameters
(u and e_dev) because they have easy expressions in terms of other known
quantities.

Args:
 - e_dev_dir: "Theta direction": this is 1 or -1, depending on whether
     increasing e_dev increases the weight of this GalaxyCacheComponent
     (1) or decreases it (-1).
 - e_dev_i: The weight given to this type of galaxy for this celestial object.
     This is either e_dev or (1 - e_dev).
 - gc: The galaxy component to be convolved
 - pc: The psf component to be convolved
 - u: The location of the celestial object in pixel coordinates as a 2x1 vector
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
        sig_sf.t .*= gc.nuBar

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
Transform the bvn derivatives and hessians from (x, sigma) to the
galaxy parameters (u, gal_shape_ids).

TODO: preallocate this memory in global variables?
""" ->
function transform_bvn_derivs{NumType <: Number}(
    bvn_sf::BvnDerivs{NumType},
    gcc::GalaxyCacheComponent{NumType},
    wcs_jacobian::Array{Float64, 2})

    # Derivatives.  Here, s stands for "shape".
    bvn_u_d = zeros(NumType, 2)
    bvn_s_d = zeros(NumType, length(gal_shape_ids))

    # The hessians.
    bvn_uu_h = zeros(NumType, 2, 2)
    bvn_ss_h = zeros(NumType, length(gal_shape_ids), length(gal_shape_ids))
    bvn_us_h = zeros(NumType, 2, length(gal_shape_ids))

    # Gradient calculations.

    # Note that dxA_duB = -wcs_jacobian[A, B].  (It is minus the jacobian
    # because the object position affects the bvn.the_mean term, which is
    # subtracted from the pixel location as defined in bvn_sf.d.)
    for x_id in 1:2, u_id in 1:2
      bvn_u_d[u_id] += -bvn_sf.d[bvn_ids.x[x_id]] * wcs_jacobian[x_id, u_id]
    end

    # Use the chain rule for the shape derviatives.
    for shape_id in 1:length(gal_shape_ids), sig_id in 1:3
      bvn_s_d[shape_id] +=
        bvn_sf.d[bvn_ids.sig[sig_id]] * gcc.sig_sf.j[sig_id, shape_id]
    end

    # Hessian calculations.

    # Second derivatives involving only u.
    # As above, dxA_duB = -wcs_jacobian[A, B] and d2x / du2 = 0.
    # TODO: eliminate the redunant term.
    for u_id1 in 1:2, u_id2 in 1:2, x_id1 in 1:2, x_id2 in 1:2
      bvn_uu_h[u_id1, u_id2] +=
        bvn_sf.h[bvn_ids.x[x_id1], bvn_ids.x[x_id2]] *
        wcs_jacobian[x_id1, u_id1] * wcs_jacobian[x_id2, u_id2]
    end

    # Second derviatives involving only shape parameters.
    # TODO: eliminate redundancies.
    for shape_id1 in 1:length(gal_shape_ids),
        shape_id2 in 1:length(gal_shape_ids)
      for sig_id1 in 1:3
        bvn_ss_h[shape_id1, shape_id2] +=
          bvn_sf.d[bvn_ids.sig[sig_id1]] *
          gcc.sig_sf.t[sig_id1, shape_id1, shape_id2]
        for sig_id2 in 1:3
          bvn_ss_h[shape_id1, shape_id2] +=
            bvn_sf.h[bvn_ids.sig[sig_id1], bvn_ids.sig[sig_id2]] *
            gcc.sig_sf.j[sig_id1, shape_id1] *
            gcc.sig_sf.j[sig_id2, shape_id2]
        end
      end
    end

    # Second derivates involving both a shape term and a u term.
    for shape_id in 1:length(gal_shape_ids), u_id in 1:2,
        sig_id in 1:3, x_id in 1:2
      bvn_us_h[u_id, shape_id] +=
        bvn_sf.h[bvn_ids.sig[sig_id], bvn_ids.x[x_id]] *
        bvn_s_d[sig_id] * bvn_u_d[u_id]
    end

    bvn_u_d, bvn_s_d, bvn_uu_h, bvn_ss_h, bvn_us_h
end
