"""
Unpack a rotation-parameterized BVN covariance matrix.

Args:
 - ab: The ratio of the minor to the major axis.
 - angle: Rotation angle (in radians)
 - scale: The major axis.

Returns:
  The 2x2 covariance matrix parameterized by the inputs.
"""
function get_bvn_cov{NumType <: Number}(
    ab::NumType, angle::NumType, scale::NumType)

   if NumType <: AbstractFloat
       @assert 0 < scale
       @assert 0 < ab <= 1.
   end

   cp = cos(angle)
   sp = sin(angle)
   ab_term = (ab ^ 2 - 1)
   scale_squared = scale ^ 2
   off_diag_term = -scale_squared * cp * sp * ab_term
   NumType[ scale_squared * (1 + ab_term * (sp ^ 2))    off_diag_term;
            off_diag_term    scale_squared * (1 + ab_term * (cp ^ 2))]
end



"""
Pre-allocated memory for quantities related to derivatives of bivariate
normals.
"""
type BivariateNormalDerivatives{NumType <: Number}

  # Pre-allocated memory for py1, py2, and f when evaluating BVNs
  py1::Array{NumType, 1}
  py2::Array{NumType, 1}
  f_pre::Array{NumType, 1}

  # Derivatives of a bvn with respect to (x, sig).
  bvn_x_d::Array{NumType, 1}
  bvn_sig_d::Array{NumType, 1}
  bvn_xx_h::Array{NumType, 2}
  bvn_xsig_h::Array{NumType, 2}
  bvn_sigsig_h::Array{NumType, 2}

  # intermediate values used in d bvn / d(x, sig)
  dpy1_dsig::Array{NumType, 1}
  dpy2_dsig::Array{NumType, 1}

  # TODO: delete this, it is now in BvnComponent
  dsiginv_dsig::Array{NumType, 2}

  # Derivatives of a bvn with respect to (u, shape)
  bvn_u_d::Array{NumType, 1}
  bvn_uu_h::Array{NumType, 2}
  bvn_s_d::Array{NumType, 1}
  bvn_ss_h::Array{NumType, 2}
  bvn_us_h::Array{NumType, 2}

  function BivariateNormalDerivatives(ThisNumType::DataType)
    py1 = zeros(ThisNumType, 1)
    py2 = zeros(ThisNumType, 1)
    f_pre = zeros(ThisNumType, 1)

    bvn_x_d = zeros(ThisNumType, 2)
    bvn_sig_d = zeros(ThisNumType, 3)
    bvn_xx_h = zeros(ThisNumType, 2, 2)
    bvn_xsig_h = zeros(ThisNumType, 2, 3)
    bvn_sigsig_h = zeros(ThisNumType, 3, 3)

    dpy1_dsig = zeros(ThisNumType, 3)
    dpy2_dsig = zeros(ThisNumType, 3)
    dsiginv_dsig = zeros(ThisNumType, 3, 3)

    # Derivatives wrt u.
    bvn_u_d = zeros(ThisNumType, 2)
    bvn_uu_h = zeros(ThisNumType, 2, 2)

    # Shape deriviatives.  Here, s stands for "shape".
    bvn_s_d = zeros(ThisNumType, length(gal_shape_ids))

    # The hessians.
    bvn_ss_h = zeros(ThisNumType, length(gal_shape_ids), length(gal_shape_ids))
    bvn_us_h = zeros(ThisNumType, 2, length(gal_shape_ids))

    new(py1, py2, f_pre,
        bvn_x_d, bvn_sig_d, bvn_xx_h, bvn_xsig_h, bvn_sigsig_h,
        dpy1_dsig, dpy2_dsig,
        dsiginv_dsig,
        bvn_u_d, bvn_uu_h, bvn_s_d, bvn_ss_h, bvn_us_h)
  end
end


"""
Relevant parameters of a bivariate normal distribution.

Args:
  the_mean: The mean as a 2x1 column vector
  the_cov: The covaraiance as a 2x2 matrix
  weight: A scalar weight

Attributes:
   the_mean: The mean argument
   precision: The inverse of the_cov
   z: The weight times the normalizing constant.
   dsiginv_dsig: The derivative of sigma inverse with respect to sigma.
   major_sd: The standard deviation of the major axis.
"""
immutable BvnComponent{NumType <: Number}
    the_mean::Vector{NumType}
    precision::Matrix{NumType}
    z::NumType
    dsiginv_dsig::Matrix{NumType}
    major_sd::NumType

    function BvnComponent{T1 <: Number, T2 <: Number, T3 <: Number}(
        the_mean::Vector{T1}, the_cov::Matrix{T2}, weight::T3;
        calculate_siginv_deriv::Bool=true)

      ThisNumType = promote_type(T1, T2, T3)
      the_det = the_cov[1,1] * the_cov[2,2] - the_cov[1,2] * the_cov[2,1]
      c = 1 ./ (the_det^.5 * 2pi)
      major_sd = sqrt(maximum([ the_cov[1, 1], the_cov[2, 2] ]))

      precision = Array(NumType, 2, 2)
      precision[2, 2] = the_cov[1,1] / the_det
      precision[1, 1] = the_cov[2,2] / the_det
      precision[1, 2] = -the_cov[1, 2] / the_det
      precision[2, 1] = -the_cov[2, 1] / the_det

      if calculate_siginv_deriv
        # Derivatives of Sigma^{-1} with respect to sigma.  These are the second
        # derivatives of log|Sigma| with respect to sigma.
        # dsiginv_dsig[a, b] is the derivative of sig^{-1}[a] / d sig[b]
        dsiginv_dsig = Array(ThisNumType, 3, 3)

        dsiginv_dsig[1, 1] = -precision[1, 1] ^ 2
        dsiginv_dsig[1, 2] = -2 * precision[1, 1] * precision[1, 2]
        dsiginv_dsig[1, 3] = -precision[1, 2] ^ 2

        dsiginv_dsig[2, 1] = -precision[1, 1] * precision[2, 1]
        dsiginv_dsig[2, 2] =
          -(precision[1, 1] * precision[2, 2] + precision[1, 2] ^ 2)
        dsiginv_dsig[2, 3] = -precision[2, 2] * precision[1, 2]

        dsiginv_dsig[3, 1] = -precision[1, 2] ^ 2
        dsiginv_dsig[3, 2] = - 2 * precision[2, 2] * precision[2, 1]
        dsiginv_dsig[3, 3] = -precision[2, 2] ^ 2
        new{ThisNumType}(the_mean, precision, c * weight,
                         dsiginv_dsig, major_sd)
      else
        new{ThisNumType}(the_mean, precision, c * weight,
                         zeros(ThisNumType, 0, 0), major_sd)
      end
    end
end


"""
Check whether a point is close enough to a BvnComponent to bother making
calculations with it.
"""
function check_point_close_to_bvn{NumType <: Number}(
    bmc::BvnComponent{NumType}, x::Vector{Float64}, num_allowed_sd::Float64)

    dist = sqrt(norm(x - bmc.the_mean))
    return dist < (num_allowed_sd * bmc.major_sd)
end


"""
Return quantities related to the pdf of an offset bivariate normal.

Args:
  - bvn_derivs: BivariateNormalDerivatives to be updated in place
  - bmc: A bivariate normal component
  - x: A 2x1 vector containing a mean offset to be applied to bmc

Returns:
  In bvn_derivs, sets these values in place:
  - py1: The first row of the precision times (x - the_mean)
  - py2: The second row of the precision times (x - the_mean)
  - The density of the bivariate normal times the weight.
"""
function eval_bvn_pdf!{NumType <: Number}(
    bvn_derivs::BivariateNormalDerivatives{NumType},
    bmc::BvnComponent{NumType}, x::Vector{Float64})

  bvn_derivs.py1[1] =
    bmc.precision[1,1] * (x[1] - bmc.the_mean[1]) +
    bmc.precision[1,2] * (x[2] - bmc.the_mean[2])
  bvn_derivs.py2[1] =
    bmc.precision[2,1] * (x[1] - bmc.the_mean[1]) +
    bmc.precision[2,2] * (x[2] - bmc.the_mean[2])
  bvn_derivs.f_pre[1] =
    bmc.z * exp(-0.5 * ((x[1] - bmc.the_mean[1]) * bvn_derivs.py1[1] +
                        (x[2] - bmc.the_mean[2]) * bvn_derivs.py2[1]))
end


##################
# Derivatives

"""
Calculate the value, gradient, and hessian of
  -0.5 * x' sigma^-1 x - 0.5 * log|sigma|
with respect to x and sigma.  This assumes that
  bvn_derivs.py1, .py2, and .f_pre have already been populated
  witha call to eval_bvn_pdf!.

Args:
  - elbo_vars: A data structure with pre-allocated intermediate variables.
  - bvn: A bivariate normal component to get derivatives for.
  - calculate_x_hess: Whether to calcualte x Hessian terms.
  - calculate_sigma_hess: Whether to calcualte sigma Hessian terms.
"""
function get_bvn_derivs!{NumType <: Number}(
    bvn_derivs::BivariateNormalDerivatives{NumType},
    bvn::BvnComponent{NumType}, calculate_x_hess::Bool,
    calculate_sigma_hessian::Bool)

  # Gradient with respect to x.
  bvn_x_d = bvn_derivs.bvn_x_d
  bvn_x_d[1] = -bvn_derivs.py1[1]
  bvn_x_d[2] = -bvn_derivs.py2[1]

  if calculate_x_hess
    bvn_xx_h = bvn_derivs.bvn_xx_h

    # Hessian terms involving only x
    bvn_xx_h[1, 1] = -bvn.precision[1, 1]
    bvn_xx_h[2, 2] = -bvn.precision[2, 2]
    bvn_xx_h[1, 2] = bvn_xx_h[2, 1] = -bvn.precision[1 ,2]
  end

  # The first term is the derivative of -0.5 * x' Sigma^{-1} x
  # The second term is the derivative of -0.5 * log|Sigma|
  bvn_sig_d = bvn_derivs.bvn_sig_d
  bvn_sig_d[1] =
    0.5 * bvn_derivs.py1[1] * bvn_derivs.py1[1] - 0.5 * bvn.precision[1, 1]
  bvn_sig_d[2] =
    bvn_derivs.py1[1] * bvn_derivs.py2[1]             - bvn.precision[1, 2]
  bvn_sig_d[3] =
    0.5 * bvn_derivs.py2[1] * bvn_derivs.py2[1] - 0.5 * bvn.precision[2, 2]

  if calculate_sigma_hessian

    # Hessian calculation for terms containing sigma.

    # Derivatives of py1 and py2 with respect to s11, s12, s22 in that order.
    # These are used for the hessian calculations.
    dpy1_dsig = bvn_derivs.dpy1_dsig
    dpy1_dsig[1] = -bvn_derivs.py1[1] * bvn.precision[1,1]
    dpy1_dsig[2] = -bvn_derivs.py2[1] * bvn.precision[1,1] -
                    bvn_derivs.py1[1] * bvn.precision[1,2]
    dpy1_dsig[3] = -bvn_derivs.py2[1] * bvn.precision[1,2]

    dpy2_dsig = bvn_derivs.dpy2_dsig
    dpy2_dsig[1] = -bvn_derivs.py1[1] * bvn.precision[1,2]
    dpy2_dsig[2] = -bvn_derivs.py1[1] * bvn.precision[2,2] -
                    bvn_derivs.py2[1] * bvn.precision[1,2]
    dpy2_dsig[3] = -bvn_derivs.py2[1] * bvn.precision[2,2]

    # Hessian terms involving only sigma
    bvn_sigsig_h = bvn_derivs.bvn_sigsig_h
    for s_ind=1:3
      # Differentiate with respect to s_ind second.
      bvn_sigsig_h[1, s_ind] = #bvn_sigsig_h[s_ind, 1] =
        bvn_derivs.py1[1] * dpy1_dsig[s_ind] - 0.5 * bvn.dsiginv_dsig[1, s_ind]

      # d log|sigma| / dsigma12 is twice lambda12.
      bvn_sigsig_h[2, s_ind] =
        bvn_derivs.py1[1] * dpy2_dsig[s_ind] + bvn_derivs.py2[1] * dpy1_dsig[s_ind] -
        bvn.dsiginv_dsig[2, s_ind]

      bvn_sigsig_h[3, s_ind] = #bvn_sigsig_h[s_ind, 3] =
        bvn_derivs.py2[1] * dpy2_dsig[s_ind] - 0.5 * bvn.dsiginv_dsig[3, s_ind]
    end

    # Hessian terms involving both x and sigma.
    # Note that dpyA / dxB = bvn.precision[A, B]
    bvn_xsig_h = bvn_derivs.bvn_xsig_h
    for x_ind=1:2
      bvn_xsig_h[x_ind, 1] = bvn_derivs.py1[1] * bvn.precision[1, x_ind]
      bvn_xsig_h[x_ind, 2] =
        bvn_derivs.py1[1] * bvn.precision[2, x_ind] +
        bvn_derivs.py2[1] * bvn.precision[1, x_ind]
      bvn_xsig_h[x_ind, 3] = bvn_derivs.py2[1] * bvn.precision[2, x_ind]
    end
  end
end




##############################

"""
The derivatives of sigma with respect to the galaxy shape parameters.  In
each case, sigma is [Sigma11, Sigma12, Sigma22], and the galaxy shape
parameters are indexed by GalaxyShapeParams.
 - j: A Jacobian matrix Sigma x GalaxyShapeParams of
      d Sigma / d GalaxyShapeParams
 - t: A Sigma x GalaxyShapeParams x GalaxyShapeParams tensor of second
      derivatives d2 Sigma / d GalaxyShapeParams d GalaxyShapeParams.
"""
type GalaxySigmaDerivs{NumType <: Number}
    j::Matrix{NumType}
    t::Array{NumType, 3}
end


"""
Args:
  - e_angle: Phi in the notes
  - e_axis: Rho in the notes
  - e_scale: Lower case sigma in the notes
  - XiXi: The value of sigma.

Note that nubar is not included.
"""
function GalaxySigmaDerivs{NumType <: Number}(
    e_angle::NumType, e_axis::NumType, e_scale::NumType,
    XiXi::Matrix{NumType}; calculate_tensor::Bool=true)

  cos_sin = cos(e_angle)sin(e_angle)
  sin_sq = sin(e_angle)^2
  cos_sq = cos(e_angle)^2

  j = Array(NumType, 3, length(gal_shape_ids))
  for i = 1:3
    j[i, gal_shape_ids.e_axis] =
      2 * e_axis * e_scale^2 * [sin_sq, -cos_sin, cos_sq][i]
    j[i, gal_shape_ids.e_angle] =
      e_scale^2 * (e_axis^2 - 1) * [2cos_sin, sin_sq - cos_sq, -2cos_sin][i]
    j[i, gal_shape_ids.e_scale] = (2XiXi ./ e_scale)[[1, 2, 4][i]]
  end

  t = Array(NumType, 3, length(gal_shape_ids), length(gal_shape_ids))
  if calculate_tensor
    # Second derivatives.

    for i = 1:3
      # Second derivatives involving e_scale
      t[i, gal_shape_ids.e_scale, gal_shape_ids.e_scale] =
        (2 * XiXi ./ (e_scale ^ 2))[[1, 2, 4][i]]
      t[i, gal_shape_ids.e_scale, gal_shape_ids.e_axis] =
        (2 * j[i, gal_shape_ids.e_axis] ./ e_scale)
      t[i, gal_shape_ids.e_scale, gal_shape_ids.e_angle] =
        (2 * j[i, gal_shape_ids.e_angle] ./ e_scale)

      t[i, gal_shape_ids.e_axis, gal_shape_ids.e_scale] =
        t[i, gal_shape_ids.e_scale, gal_shape_ids.e_axis]
      t[i, gal_shape_ids.e_angle, gal_shape_ids.e_scale] =
        t[i, gal_shape_ids.e_scale, gal_shape_ids.e_angle]

      # Remaining second derivatives involving e_angle
      t[i, gal_shape_ids.e_angle, gal_shape_ids.e_angle] =
        2 * e_scale^2 * (e_axis^2 - 1) *
        [cos_sq - sin_sq, 2cos_sin, sin_sq - cos_sq][i]
      t[i, gal_shape_ids.e_angle, gal_shape_ids.e_axis] =
        2 * e_scale^2 * e_axis * [2cos_sin, sin_sq - cos_sq, -2cos_sin][i]
      t[i, gal_shape_ids.e_axis, gal_shape_ids.e_angle] =
        t[i, gal_shape_ids.e_angle, gal_shape_ids.e_axis]

      # The second derivative involving only e_axis.
      t[i, gal_shape_ids.e_axis, gal_shape_ids.e_axis] =
        2 * e_scale^2 * [sin_sq, -cos_sin, cos_sq][i]
    end
  else
    fill!(t, 0.0)
  end

  GalaxySigmaDerivs(j, t)
end


"""
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
"""
immutable GalaxyCacheComponent{NumType <: Number}
    e_dev_dir::Float64
    e_dev_i::NumType
    bmc::BvnComponent{NumType}
    sig_sf::GalaxySigmaDerivs{NumType}
    # [Sigma11, Sigma12, Sigma22] x [e_axis, e_angle, e_scale]
end


function GalaxyCacheComponent{NumType <: Number}(
    e_dev_dir::Float64, e_dev_i::NumType,
    gc::GalaxyComponent, pc::PsfComponent, u::Vector{NumType},
    e_axis::NumType, e_angle::NumType, e_scale::NumType,
    calculate_derivs::Bool, calculate_hessian::Bool)

  # Declare in advance to save memory allocation.
  const empty_sig_sf =
    GalaxySigmaDerivs(Array(NumType, 0, 0), Array(NumType, 0, 0, 0))

  XiXi = get_bvn_cov(e_axis, e_angle, e_scale)
  mean_s = NumType[pc.xiBar[1] + u[1], pc.xiBar[2] + u[2]]
  var_s = pc.tauBar + gc.nuBar * XiXi
  weight = pc.alphaBar * gc.etaBar  # excludes e_dev

  # d siginv / dsigma is only necessary for the Hessian.
  bmc = BvnComponent{NumType}(
    mean_s, var_s, weight,
    calculate_siginv_deriv=calculate_derivs && calculate_hessian)

  if calculate_derivs
    sig_sf = GalaxySigmaDerivs(
      e_angle, e_axis, e_scale, XiXi, calculate_tensor=calculate_hessian)
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


###################################################
# Transform derivatives into the model parameterization.

"""
Transform the bvn derivatives and hessians from (x) to the
galaxy parameters (u). Updates bvn_u_d and bvn_uu_h in place.

bvn_x_d and bvn_xx_h should already have been set using get_bvn_derivs!()
"""
function transform_bvn_ux_derivs!{NumType <: Number}(
    bvn_derivs::BivariateNormalDerivatives{NumType},
    wcs_jacobian::Array{Float64, 2}, calculate_hessian::Bool)

  # Gradient calculations.

  # Note that dxA_duB = -wcs_jacobian[A, B].  (It is minus the jacobian
  # because the object position affects the bvn.the_mean term, which is
  # subtracted from the pixel location as defined in bvn_sf.d.)
  bvn_u_d = bvn_derivs.bvn_u_d
  bvn_x_d = bvn_derivs.bvn_x_d
  bvn_u_d[1] =
    -(bvn_x_d[1] * wcs_jacobian[1, 1] + bvn_x_d[2] * wcs_jacobian[2, 1])
  bvn_u_d[2] =
    -(bvn_x_d[1] * wcs_jacobian[1, 2] + bvn_x_d[2] * wcs_jacobian[2, 2])

  if calculate_hessian
    # Hessian calculations.

    bvn_uu_h = bvn_derivs.bvn_uu_h
    bvn_xx_h = bvn_derivs.bvn_xx_h
    fill!(bvn_uu_h, 0.0)
    # Second derivatives involving only u.
    # As above, dxA_duB = -wcs_jacobian[A, B] and d2x / du2 = 0.
    # TODO: time consuming **************
    @inbounds for x_id2 in 1:2, x_id1 in 1:2, u_id2 in 1:2
      inner_term = bvn_xx_h[x_id1, x_id2] * wcs_jacobian[x_id2, u_id2]
      @inbounds for u_id1 in 1:u_id2
        bvn_uu_h[u_id1, u_id2] += inner_term * wcs_jacobian[x_id1, u_id1]
      end
    end
    @inbounds bvn_uu_h[2, 1] = bvn_uu_h[1, 2]
  end
end


"""
Transform all the bvn derivatives from x and sigma to u and the model
parameters.

These values should already have been set using get_bvn_derivs!():
bvn_x_d
bvn_xx_h
bvn_sig_d
bvn_sigsig_h
bvn_xsig_h
"""
function transform_bvn_derivs!{NumType <: Number}(
    bvn_derivs::BivariateNormalDerivatives{NumType},
    sig_sf::GalaxySigmaDerivs{NumType},
    wcs_jacobian::Array{Float64, 2}, calculate_hessian::Bool)

  # Transform the u derivates first.
  # bvn_x_d and bvn_xx_h should already have been set using get_bvn_derivs!()
  transform_bvn_ux_derivs!(bvn_derivs, wcs_jacobian, calculate_hessian)

  # Gradient calculations.

  # Use the chain rule for the shape derviatives.
  # TODO: time consuming **************
  bvn_s_d = bvn_derivs.bvn_s_d
  bvn_sig_d = bvn_derivs.bvn_sig_d

  fill!(bvn_s_d, 0.0)
  @inbounds for shape_id in 1:length(gal_shape_ids), sig_id in 1:3
    bvn_s_d[shape_id] += bvn_sig_d[sig_id] * sig_sf.j[sig_id, shape_id]
  end

  if calculate_hessian
    # Hessian calculations.

    bvn_ss_h = bvn_derivs.bvn_ss_h
    bvn_us_h = bvn_derivs.bvn_us_h

    fill!(bvn_ss_h, 0.0)
    fill!(bvn_us_h, 0.0)

    # Second derviatives involving only shape parameters.
    # TODO: time consuming **************
    @inbounds for shape_id2 in 1:length(gal_shape_ids), shape_id1 in 1:shape_id2
      @inbounds for sig_id1 in 1:3
        bvn_ss_h[shape_id1, shape_id2] +=
          bvn_sig_d[sig_id1] * sig_sf.t[sig_id1, shape_id1, shape_id2]
      end
    end

    bvn_sigsig_h = bvn_derivs.bvn_sigsig_h
    @inbounds for sig_id1 in 1:3, sig_id2 in 1:3,
                  shape_id2 in 1:length(gal_shape_ids)
      inner_term =
        bvn_sigsig_h[sig_id1, sig_id2] * sig_sf.j[sig_id2, shape_id2]
      @inbounds for shape_id1 in 1:shape_id2
        bvn_ss_h[shape_id1, shape_id2] +=
          inner_term * sig_sf.j[sig_id1, shape_id1]
      end
    end

    @inbounds for shape_id2 in 1:length(gal_shape_ids), shape_id1 in 1:shape_id2
      bvn_ss_h[shape_id2, shape_id1] = bvn_ss_h[shape_id1, shape_id2]
    end

    # Second derivates involving both a shape term and a u term.
    # TODO: time consuming **************
    bvn_xsig_h = bvn_derivs.bvn_xsig_h
    @inbounds for shape_id in 1:length(gal_shape_ids),
                  u_id in 1:2, sig_id in 1:3, x_id in 1:2
      bvn_us_h[u_id, shape_id] +=
        bvn_xsig_h[x_id, sig_id] * sig_sf.j[sig_id, shape_id] *
        (-wcs_jacobian[x_id, u_id])
    end
  end
end
