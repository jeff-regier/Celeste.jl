# Defining gal_shape_ids_len might not be needed but it is critical that this is compile time constant
const gal_shape_ids_len = 3

using Celeste: Const, @aliasscope, @unroll_loop

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
    @SMatrix NumType[scale_squared * (1 + ab_term * (sp ^ 2))  off_diag_term;
                     off_diag_term                             scale_squared * (1 + ab_term * (cp ^ 2))]
end



"""
Pre-allocated memory for quantities related to derivatives of bivariate
normals.
"""
type BivariateNormalDerivatives{NumType <: Number}

  # Pre-allocated memory for py1, py2, and f when evaluating BVNs
  py1::NumType
  py2::NumType
  f_pre::NumType

  # Derivatives of a bvn with respect to (x, sig).
  bvn_x_d::SVector{2, NumType}
  bvn_sig_d::SVector{3, NumType}
  bvn_xx_h::SArray{Tuple{2,2}, NumType, 2, 4}
  bvn_xsig_h::SArray{Tuple{2,3}, NumType, 2, 6}
  bvn_sigsig_h::SArray{Tuple{3,3}, NumType, 2, 9}

  # Derivatives of a bvn with respect to (u, shape)
  bvn_u_d::SVector{2, NumType}
  bvn_uu_h::SMatrix{2, 2, NumType, 4}
  bvn_s_d::SVector{length(gal_shape_ids), NumType}
  bvn_ss_h::SArray{Tuple{length(gal_shape_ids), length(gal_shape_ids)}, NumType, 2, length(gal_shape_ids)^2}
  bvn_us_h::SArray{Tuple{2, length(gal_shape_ids)}, NumType, 2, 2*length(gal_shape_ids)}

  function (::Type{BivariateNormalDerivatives{NumType}}){NumType}()
    py1 = zero(NumType)
    py2 = zero(NumType)
    f_pre = zero(NumType)

    bvn_x_d = zeros(NumType, 2)
    bvn_sig_d = zeros(NumType, 3)
    bvn_xx_h = zeros(NumType, 2, 2)
    bvn_xsig_h = zeros(NumType, 2, 3)
    bvn_sigsig_h = zeros(NumType, 3, 3)

    # Derivatives wrt u.
    bvn_u_d = zeros(NumType, 2)
    bvn_uu_h = zeros(NumType, 2, 2)

    # Shape deriviatives.  Here, s stands for "shape".
    bvn_s_d = zeros(NumType, length(gal_shape_ids))

    # The hessians.
    bvn_ss_h = zeros(NumType, length(gal_shape_ids), length(gal_shape_ids))
    bvn_us_h = zeros(NumType, 2, length(gal_shape_ids))

    new{NumType}(py1, py2, f_pre,
        bvn_x_d, bvn_sig_d, bvn_xx_h, bvn_xsig_h, bvn_sigsig_h,
        bvn_u_d, bvn_uu_h, bvn_s_d, bvn_ss_h, bvn_us_h)
  end
end

function clear!{T}(bvn_derivs::BivariateNormalDerivatives{T})
    x = zero(T)
    bvn_derivs.py1 = x
    bvn_derivs.py2 = x
    bvn_derivs.f_pre = x
    bvn_derivs.bvn_x_d = fill(x, typeof(bvn_derivs.bvn_x_d))
    bvn_derivs.bvn_sig_d = fill(x, typeof(bvn_derivs.bvn_sig_d))
    bvn_derivs.bvn_xx_h = fill(x, typeof(bvn_derivs.bvn_xx_h))
    bvn_derivs.bvn_xsig_h = fill(x, typeof(bvn_derivs.bvn_xsig_h))
    bvn_derivs.bvn_sigsig_h = fill(x, typeof(bvn_derivs.bvn_sigsig_h))
    bvn_derivs.bvn_u_d = fill(x, typeof(bvn_derivs.bvn_u_d))
    bvn_derivs.bvn_uu_h = fill(x, typeof(bvn_derivs.bvn_uu_h))
    bvn_derivs.bvn_s_d = fill(x, typeof(bvn_derivs.bvn_s_d))
    bvn_derivs.bvn_ss_h = fill(x, typeof(bvn_derivs.bvn_ss_h))
    bvn_derivs.bvn_us_h = fill(x, typeof(bvn_derivs.bvn_us_h))
    return bvn_derivs
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
    the_mean::SVector{2,NumType}
    precision::SMatrix{2,2,NumType,4}
    z::NumType
    dsiginv_dsig::SMatrix{3,3,NumType,9}
    major_sd::NumType
end

function BvnComponent{T1<:Number,T2<:Number,T3<:Number}(
    the_mean::SVector{2,T1}, the_cov::SMatrix{2,2,T2,4}, weight::T3,
    calculate_siginv_deriv::Bool=true)

    NumType = promote_type(T1,T2,T3)
    c = 1 / (sqrt(det(the_cov)) * 2pi)
    major_sd = sqrt(max( the_cov[1, 1], the_cov[2, 2] ))

    precision = inv(the_cov)

    if calculate_siginv_deriv
        # Derivatives of Sigma^{-1} with respect to sigma.  These are the second
        # derivatives of log|Sigma| with respect to sigma.
        # dsiginv_dsig[a, b] is the derivative of sig^{-1}[a] / d sig[b]

        dsiginv_dsig11 = -precision[1, 1] ^ 2
        dsiginv_dsig12 = -2 * precision[1, 1] * precision[1, 2]
        dsiginv_dsig13 = -precision[1, 2] ^ 2

        dsiginv_dsig21 = -precision[1, 1] * precision[2, 1]
        dsiginv_dsig22 =
          -(precision[1, 1] * precision[2, 2] + precision[1, 2] ^ 2)
        dsiginv_dsig23 = -precision[2, 2] * precision[1, 2]

        dsiginv_dsig31 = -precision[1, 2] ^ 2
        dsiginv_dsig32 = - 2 * precision[2, 2] * precision[2, 1]
        dsiginv_dsig33 = -precision[2, 2] ^ 2

        dsiginv_dsig = @SMatrix NumType[ dsiginv_dsig11 dsiginv_dsig12 dsiginv_dsig13;
                                           dsiginv_dsig21 dsiginv_dsig22 dsiginv_dsig23;
                                           dsiginv_dsig13 dsiginv_dsig32 dsiginv_dsig33 ]

        BvnComponent{NumType}(the_mean, precision, c * weight,
                              dsiginv_dsig, major_sd)
    else
        BvnComponent{NumType}(the_mean, precision, c * weight,
                              zeros(SMatrix{3,3,NumType,9}), major_sd)
    end
end

using Core.Intrinsics: llvmcall

# Use a form of exp that llvm can recognize and vectorize
function llvm_exp(x::Float64)
    llvmcall(
        ("""declare double @llvm.exp.f64(double)""",
         """%2 = call double @llvm.exp.f64(double %0)
            ret double %2"""),
    Float64, Tuple{Float64}, x)
end
llvm_exp(x) = exp(x)

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
    bmc::BvnComponent{NumType}, x::SVector{2,Float64})

  bvn_derivs.py1 =
    bmc.precision[1,1] * (x[1] - bmc.the_mean[1]) +
    bmc.precision[1,2] * (x[2] - bmc.the_mean[2])
  bvn_derivs.py2 =
    bmc.precision[2,1] * (x[1] - bmc.the_mean[1]) +
    bmc.precision[2,2] * (x[2] - bmc.the_mean[2])
  bvn_derivs.f_pre =
    bmc.z * llvm_exp(-0.5 * ((x[1] - bmc.the_mean[1]) * bvn_derivs.py1 +
                             (x[2] - bmc.the_mean[2]) * bvn_derivs.py2))
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

  @inbounds begin

    # Gradient with respect to x.
    bvn_derivs.bvn_x_d = -(@SVector [bvn_derivs.py1, bvn_derivs.py2])

    if calculate_x_hess
      bvn_derivs.bvn_xx_h = -bvn.precision
    end

    # The first term is the derivative of -0.5 * x' Sigma^{-1} x
    # The second term is the derivative of -0.5 * log|Sigma|
    bvn_derivs.bvn_sig_d = @SVector NumType[
      0.5 * bvn_derivs.py1 * bvn_derivs.py1 - 0.5 * bvn.precision[1, 1],
      bvn_derivs.py1 * bvn_derivs.py2             - bvn.precision[1, 2],
      0.5 * bvn_derivs.py2 * bvn_derivs.py2 - 0.5 * bvn.precision[2, 2]
    ]

    if calculate_sigma_hessian

      # Hessian calculation for terms containing sigma.

      # Derivatives of py1 and py2 with respect to s11, s12, s22 in that order.
      # These are used for the hessian calculations.
      dpy1_dsig = @SVector NumType[
        -bvn_derivs.py1 * bvn.precision[1,1],
        -bvn_derivs.py2 * bvn.precision[1,1] - bvn_derivs.py1 * bvn.precision[1,2],
        -bvn_derivs.py2 * bvn.precision[1,2]
      ]

      dpy2_dsig = @SVector NumType[
        -bvn_derivs.py1 * bvn.precision[1,2],
        -bvn_derivs.py1 * bvn.precision[2,2] - bvn_derivs.py2 * bvn.precision[1,2],
        -bvn_derivs.py2 * bvn.precision[2,2]
      ]

      # Hessian terms involving only sigma
      bvn_derivs.bvn_sigsig_h = vcat(
        # Differentiate with respect to s_ind second.
        (bvn_derivs.py1[1] * dpy1_dsig - 0.5 * bvn.dsiginv_dsig[1, :])',
        # d log|sigma| / dsigma12 is twice lambda12.
        (bvn_derivs.py1[1] * dpy2_dsig + bvn_derivs.py2[1] * dpy1_dsig -
                bvn.dsiginv_dsig[2, :])',
        (bvn_derivs.py2[1] * dpy2_dsig - 0.5 * bvn.dsiginv_dsig[3, :])'
      )

      # Hessian terms involving both x and sigma.
      # Note that dpyA / dxB = bvn.precision[A, B]
      bvn_derivs.bvn_xsig_h = hcat(
        (bvn_derivs.py1[1] * bvn.precision[1, :]),
        (bvn_derivs.py1[1] * bvn.precision[2, :] +
         bvn_derivs.py2[1] * bvn.precision[1, :]),
        (bvn_derivs.py2[1] * bvn.precision[2, :])
      )
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
immutable GalaxySigmaDerivs{NumType <: Number}
    j::SMatrix{3,gal_shape_ids_len,NumType,9}
    t::SArray{Tuple{3,gal_shape_ids_len,gal_shape_ids_len},NumType,3,27}
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
    XiXi::SMatrix{2,2,NumType,4}, nuBar::Float64=1.0, calculate_tensor::Bool=true)

  cos_sin = cos(e_angle)sin(e_angle)
  sin_sq  = sin(e_angle)^2
  cos_sq  = cos(e_angle)^2

  j = hcat(2 * e_axis * e_scale^2 * SVector{3,NumType}(sin_sq, -cos_sin, cos_sq),
           e_scale^2 * (e_axis^2 - 1) * SVector{3,NumType}(2cos_sin, sin_sq - cos_sq, -2cos_sin),
           2 * SVector{3,NumType}(XiXi[1], XiXi[2], XiXi[4]) / e_scale)

  if calculate_tensor
    # Second derivatives.

    gal_shape_ids_e_angle = Base.to_index(typeof(gal_shape_ids), gal_shape_ids.e_angle)
    gal_shape_ids_e_axis = Base.to_index(typeof(gal_shape_ids), gal_shape_ids.e_axis)

    t = SArray{Tuple{3,3,3}, NumType, 3, 27}(sin_sq * 2 * e_scale^2, -cos_sin * 2 * e_scale^2, cos_sq * 2 * e_scale^2,
      2cos_sin * 2 * e_scale^2 * e_axis, (sin_sq - cos_sq) * 2 * e_scale^2 * e_axis, -2cos_sin * 2 * e_scale^2 * e_axis,
      2 * j[1, gal_shape_ids_e_axis]  / e_scale, 2 * j[2, gal_shape_ids_e_axis]  / e_scale, 2 * j[3, gal_shape_ids_e_axis]  / e_scale,
      2cos_sin * 2 * e_scale^2 * e_axis, (sin_sq - cos_sq) * 2 * e_scale^2 * e_axis, -2cos_sin * 2 * e_scale^2 * e_axis,
      (cos_sq - sin_sq) * 2 * e_scale^2 * (e_axis^2 - 1), 2cos_sin * 2 * e_scale^2 * (e_axis^2 - 1), (sin_sq - cos_sq) * 2 * e_scale^2 * (e_axis^2 - 1),
      2 * j[1, gal_shape_ids_e_angle] / e_scale, 2 * j[2, gal_shape_ids_e_angle] / e_scale, 2 * j[3, gal_shape_ids_e_angle] / e_scale,
      2 * j[1, gal_shape_ids_e_axis]  / e_scale, 2 * j[2, gal_shape_ids_e_axis]  / e_scale, 2 * j[3, gal_shape_ids_e_axis]  / e_scale,
      2 * j[1, gal_shape_ids_e_angle] / e_scale, 2 * j[2, gal_shape_ids_e_angle] / e_scale, 2 * j[3, gal_shape_ids_e_angle] / e_scale,
      2 * XiXi[1 << (1 - 1)] / e_scale^2, 2 * XiXi[1 << (2 - 1)] / e_scale^2, 2 * XiXi[1 << (3 - 1)] / e_scale^2)

  else
    t = @SArray zeros(NumType, 3, 3, 3)
  end

  GalaxySigmaDerivs(j*nuBar, t*nuBar)
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
    gc::GalaxyComponent, pc::PsfComponent, u::AbstractVector{NumType},
    e_axis::NumType, e_angle::NumType, e_scale::NumType,
    calculate_gradient::Bool, calculate_hessian::Bool)

  XiXi = get_bvn_cov(e_axis, e_angle, e_scale)
  mean_s = @SVector NumType[pc.xiBar[1] + u[1], pc.xiBar[2] + u[2]]
  var_s = pc.tauBar + gc.nuBar * XiXi
  weight = pc.alphaBar * gc.etaBar  # excludes e_dev

  # d siginv / dsigma is only necessary for the Hessian.
  bmc = BvnComponent(mean_s, var_s, weight, calculate_gradient && calculate_hessian)

  if calculate_gradient
    sig_sf = GalaxySigmaDerivs(
      e_angle, e_axis, e_scale, XiXi, gc.nuBar, calculate_hessian)
  else
    sig_sf = GalaxySigmaDerivs(NumType)
  end

  GalaxyCacheComponent(e_dev_dir, e_dev_i, bmc, sig_sf)
end

GalaxySigmaDerivs{NumType}(::Type{NumType}) = GalaxySigmaDerivs(
                                                     @SMatrix(zeros(NumType,3,gal_shape_ids_len)),
                                                     @SArray( zeros(NumType,3,gal_shape_ids_len,gal_shape_ids_len)))

###################################################
# Transform derivatives into the model parameterization.

"""
Transform the bvn derivatives and hessians from (x) to the
galaxy parameters (u). Updates bvn_u_d and bvn_uu_h in place.

bvn_x_d and bvn_xx_h should already have been set using get_bvn_derivs!()
"""
function transform_bvn_ux_derivs!{NumType <: Number}(
    bvn_derivs::BivariateNormalDerivatives{NumType},
    wcs_jacobian, calculate_hessian::Bool)

  # Gradient calculations.

  # Note that dxA_duB = -wcs_jacobian[A, B].  (It is minus the jacobian
  # because the object position affects the bvn.the_mean term, which is
  # subtracted from the pixel location as defined in bvn_sf.d.)
  bvn_derivs.bvn_u_d = - (wcs_jacobian' * bvn_derivs.bvn_x_d)

  if calculate_hessian
    bvn_derivs.bvn_uu_h = wcs_jacobian' * bvn_derivs.bvn_xx_h * wcs_jacobian
  end
end

# WARNING: HUGE PERFORMANCE HOTSPOT
@Base.hotspot function transform_bvn_derivs_hessian!{NumType <: Number}(
    bvn_derivs::BivariateNormalDerivatives{NumType},
    sig_sf::GalaxySigmaDerivs{NumType},
    wcs_jacobian)
  @aliasscope begin
    # Hessian calculations.
    bvn_derivs.bvn_ss_h = @SMatrix zeros(Float64, 3, 3)

    # Second derviatives involving only shape parameters.
    # TODO: time consuming **************
    sig_sf_t = sig_sf.t
    for sig_id1 in 1:3
        bvn_derivs.bvn_ss_h += bvn_derivs.bvn_sig_d[sig_id1] * sig_sf.t[sig_id1, :, :]
    end

    bvn_derivs.bvn_ss_h += sig_sf.j' * bvn_derivs.bvn_sigsig_h * sig_sf.j

    # Second derivates involving both a shape term and a u term.
    # TODO: time consuming **************
    bvn_derivs.bvn_us_h = -wcs_jacobian' * bvn_derivs.bvn_xsig_h * sig_sf.j
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
    wcs_jacobian, calculate_hessian::Bool)

  # Transform the u derivates first.
  # bvn_x_d and bvn_xx_h should already have been set using get_bvn_derivs!()
  transform_bvn_ux_derivs!(bvn_derivs, wcs_jacobian, calculate_hessian)

  # Gradient calculations.
  # Use the chain rule for the shape derviatives.
  # TODO: time consuming **************
  bvn_derivs.bvn_s_d = sig_sf.j' * bvn_derivs.bvn_sig_d

  if calculate_hessian
    transform_bvn_derivs_hessian!(bvn_derivs, sig_sf, wcs_jacobian)
  end
  
  nothing
end
