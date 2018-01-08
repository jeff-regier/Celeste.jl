"""BivarateNormals

2-d gaussians and their derivatives.
"""
module BivariateNormals

using StaticArrays
using ..Celeste: Const, @aliasscope, @unroll_loop

# TODO: Many parameter and type names sound specific to "galaxies". However,
# they're actually general to bivariate normals or mixtures thereof.
# Parameters should be renamed to reflect this.

# Hardcoding this might not be needed but it is critical that this is
# compile time constant.
const GAL_SHAPE_IDS_LENGTH = 3

"""
Unpack a rotation-parameterized BVN covariance matrix.

Args:
 - ab: The ratio of the minor to the major axis.
 - angle: Rotation angle (in radians)
 - scale: The major axis.

Returns:
  The 2x2 covariance matrix parameterized by the inputs.
"""
function get_bvn_cov(ab::T, angle::T, scale::T) where {T<:Number}

    if T <: AbstractFloat
        @assert 0 < scale
        @assert 0 < ab <= 1.
    end

    cp = cos(angle)
    sp = sin(angle)
    ab_term = (ab ^ 2 - 1)
    scale_squared = scale ^ 2
    off_diag_term = -scale_squared * cp * sp * ab_term
    @SMatrix T[scale_squared * (1 + ab_term * (sp ^ 2))  off_diag_term;
               off_diag_term                             scale_squared * (1 + ab_term * (cp ^ 2))]
end



"""
Pre-allocated memory for quantities related to derivatives of bivariate
normals.
"""
struct BivariateNormalDerivatives{T<:Number}

    # Pre-allocated memory for py1, py2, and f when evaluating BVNs
    py1::Array{T, 1}
    py2::Array{T, 1}
    f_pre::Array{T, 1}

    # Derivatives of a bvn with respect to (x, sig).
    bvn_x_d::Array{T, 1}
    bvn_sig_d::Array{T, 1}
    bvn_xx_h::SizedMatrix{2, 2, T, 2}
    bvn_xsig_h::SizedMatrix{2, 3, T, 2}
    bvn_sigsig_h::SizedMatrix{3, 3, T, 2}

    # intermediate values used in d bvn / d(x, sig)
    dpy1_dsig::Array{T, 1}
    dpy2_dsig::Array{T, 1}

    # Derivatives of a bvn with respect to (pos, shape)
    bvn_u_d::Array{T, 1}
    bvn_uu_h::SizedMatrix{2, 2, T, 2}
    bvn_s_d::Array{T, 1}
    bvn_ss_h::SizedMatrix{GAL_SHAPE_IDS_LENGTH, GAL_SHAPE_IDS_LENGTH, T, 2}
    bvn_us_h::SizedMatrix{2, GAL_SHAPE_IDS_LENGTH, T, 2}

    function BivariateNormalDerivatives{T}() where {T<:Number}
        py1 = zeros(T, 1)
        py2 = zeros(T, 1)
        f_pre = zeros(T, 1)

        bvn_x_d = zeros(T, 2)
        bvn_sig_d = zeros(T, 3)
        bvn_xx_h = zeros(T, 2, 2)
        bvn_xsig_h = zeros(T, 2, 3)
        bvn_sigsig_h = zeros(T, 3, 3)

        dpy1_dsig = zeros(T, 3)
        dpy2_dsig = zeros(T, 3)

        # Derivatives wrt pos.
        bvn_u_d = zeros(T, 2)
        bvn_uu_h = zeros(T, 2, 2)

        # Shape deriviatives.  Here, s stands for "shape".
        bvn_s_d = zeros(T, GAL_SHAPE_IDS_LENGTH)

        # The hessians.
        bvn_ss_h = zeros(T, GAL_SHAPE_IDS_LENGTH, GAL_SHAPE_IDS_LENGTH)
        bvn_us_h = zeros(T, 2, GAL_SHAPE_IDS_LENGTH)

        new(py1, py2, f_pre,
            bvn_x_d, bvn_sig_d, bvn_xx_h, bvn_xsig_h, bvn_sigsig_h,
            dpy1_dsig, dpy2_dsig,
            bvn_u_d, bvn_uu_h, bvn_s_d, bvn_ss_h, bvn_us_h)
    end
end

function zero!(bvn_derivs::BivariateNormalDerivatives{T}) where {T}
    x = zero(T)
    fill!(bvn_derivs.py1, x)
    fill!(bvn_derivs.py2, x)
    fill!(bvn_derivs.f_pre, x)
    fill!(bvn_derivs.bvn_x_d, x)
    fill!(bvn_derivs.bvn_sig_d, x)
    fill!(bvn_derivs.bvn_xx_h, x)
    fill!(bvn_derivs.bvn_xsig_h, x)
    fill!(bvn_derivs.bvn_sigsig_h, x)
    fill!(bvn_derivs.dpy1_dsig, x)
    fill!(bvn_derivs.dpy2_dsig, x)
    fill!(bvn_derivs.bvn_u_d, x)
    fill!(bvn_derivs.bvn_uu_h, x)
    fill!(bvn_derivs.bvn_s_d, x)
    fill!(bvn_derivs.bvn_ss_h, x)
    fill!(bvn_derivs.bvn_us_h, x)
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
struct BvnComponent{T<:Number}
    the_mean::SVector{2,T}
    precision::SMatrix{2,2,T,4}
    z::T
    dsiginv_dsig::SMatrix{3,3,T,9}
    major_sd::T
end

function BvnComponent(
    the_mean::SVector{2,T1},
    the_cov::SMatrix{2,2,T2,4},
    weight::T3,
    calculate_siginv_deriv::Bool=true) where {T1<:Number,T2<:Number,T3<:Number}

    T = promote_type(T1, T2, T3)
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

        dsiginv_dsig = @SMatrix T[dsiginv_dsig11 dsiginv_dsig12 dsiginv_dsig13;
                                  dsiginv_dsig21 dsiginv_dsig22 dsiginv_dsig23;
                                  dsiginv_dsig13 dsiginv_dsig32 dsiginv_dsig33]

        BvnComponent{T}(the_mean, precision, c * weight,
                        dsiginv_dsig, major_sd)
    else
        BvnComponent{T}(the_mean, precision, c * weight,
                        zeros(SMatrix{3,3,T,9}), major_sd)
    end
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
function eval_bvn_pdf!(
    bvn_derivs::BivariateNormalDerivatives{T},
    bmc::BvnComponent{T},
    x::SVector{2,Float64}) where {T<:Number}

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
function get_bvn_derivs!(
    bvn_derivs::BivariateNormalDerivatives{T},
    bvn::BvnComponent{T},
    calculate_x_hess::Bool,
    calculate_sigma_hessian::Bool) where {T<:Number}

    @inbounds begin

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

            # Derivatives of py1 and py2 with respect to s11, s12, s22
            # in that order.  These are used for the hessian
            # calculations.
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
end


"""
The derivatives of sigma with respect to the galaxy shape parameters.  In
each case, sigma is [Sigma11, Sigma12, Sigma22], and the galaxy shape
parameters are indexed by GalaxyShapeParams.
 - j: A Jacobian matrix Sigma x GalaxyShapeParams of
      d Sigma / d GalaxyShapeParams
 - t: A Sigma x GalaxyShapeParams x GalaxyShapeParams tensor of second
      derivatives d2 Sigma / d GalaxyShapeParams d GalaxyShapeParams.
"""
struct GalaxySigmaDerivs{T <: Number}
    j::SMatrix{3,GAL_SHAPE_IDS_LENGTH,T,9}
    t::SArray{Tuple{3,GAL_SHAPE_IDS_LENGTH,GAL_SHAPE_IDS_LENGTH},T,3,27}
end


"""
Args:
  - gal_angle: Phi in the notes
  - gal_axis_ratio: Rho in the notes
  - gal_radius_px: Lower case sigma in the notes
  - XiXi: The value of sigma.

Note that nubar is not included.
"""
function GalaxySigmaDerivs(
    gal_angle::T,
    gal_axis_ratio::T,
    gal_radius_px::T,
    XiXi::SMatrix{2,2,T,4},
    nuBar::Float64=1.0,
    calculate_tensor::Bool=true) where {T<:Number}

    cos_sin = cos(gal_angle)sin(gal_angle)
    sin_sq  = sin(gal_angle)^2
    cos_sq  = cos(gal_angle)^2

    j = hcat(2 * gal_axis_ratio * gal_radius_px^2 * SVector{3,T}(sin_sq, -cos_sin, cos_sq),
             gal_radius_px^2 * (gal_axis_ratio^2 - 1) * SVector{3,T}(2cos_sin, sin_sq - cos_sq, -2cos_sin),
             2 * SVector{3,T}(XiXi[1], XiXi[2], XiXi[4]) / gal_radius_px)

    if calculate_tensor
        # Second derivatives.
        t = SArray{Tuple{3,3,3}, T, 3, 27}(sin_sq * 2 * gal_radius_px^2,
                                           -cos_sin * 2 * gal_radius_px^2,
                                           cos_sq * 2 * gal_radius_px^2,
                                           2cos_sin * 2 * gal_radius_px^2 * gal_axis_ratio,
                                           (sin_sq - cos_sq) * 2 * gal_radius_px^2 * gal_axis_ratio,
                                           -2cos_sin * 2 * gal_radius_px^2 * gal_axis_ratio,
                                           2 * j[1, 1]  / gal_radius_px,
                                           2 * j[2, 1]  / gal_radius_px,
                                           2 * j[3, 1]  / gal_radius_px,
                                           2cos_sin * 2 * gal_radius_px^2 * gal_axis_ratio,
                                           (sin_sq - cos_sq) * 2 * gal_radius_px^2 * gal_axis_ratio,
                                           -2cos_sin * 2 * gal_radius_px^2 * gal_axis_ratio,
                                           (cos_sq - sin_sq) * 2 * gal_radius_px^2 * (gal_axis_ratio^2 - 1),
                                           2cos_sin * 2 * gal_radius_px^2 * (gal_axis_ratio^2 - 1),
                                           (sin_sq - cos_sq) * 2 * gal_radius_px^2 * (gal_axis_ratio^2 - 1),
                                           2 * j[1, 2] / gal_radius_px,
                                           2 * j[2, 2] / gal_radius_px,
                                           2 * j[3, 2] / gal_radius_px,
                                           2 * j[1, 1]  / gal_radius_px,
                                           2 * j[2, 1]  / gal_radius_px,
                                           2 * j[3, 1]  / gal_radius_px,
                                           2 * j[1, 2] / gal_radius_px,
                                           2 * j[2, 2] / gal_radius_px,
                                           2 * j[3, 2] / gal_radius_px,
                                           2 * XiXi[1 << (1 - 1)] / gal_radius_px^2,
                                           2 * XiXi[1 << (2 - 1)] / gal_radius_px^2,
                                           2 * XiXi[1 << (3 - 1)] / gal_radius_px^2)

    else
        t = @SArray zeros(T, 3, 3, 3)
    end

    return GalaxySigmaDerivs{T}(j*nuBar, t*nuBar)
end


GalaxySigmaDerivs(::Type{T}) where {T<:Number} =
    GalaxySigmaDerivs{T}(@SMatrix(zeros(T,3,GAL_SHAPE_IDS_LENGTH)),
                         @SArray( zeros(T,3,GAL_SHAPE_IDS_LENGTH,GAL_SHAPE_IDS_LENGTH)))


###################################################
# Transform derivatives into the model parameterization.

"""
Transform the bvn derivatives and hessians from (x) to the
galaxy parameters (pos). Updates bvn_u_d and bvn_uu_h in place.

bvn_x_d and bvn_xx_h should already have been set using get_bvn_derivs!()
"""
function transform_bvn_ux_derivs!(
    bvn_derivs::BivariateNormalDerivatives{T},
    wcs_jacobian,
    calculate_hessian::Bool) where {T<:Number}

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
        # Second derivatives involving only pos.
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


@generated function fast_fill!(s::T, x) where {T<:SizedMatrix}
    quote
        $(Expr(:meta, :inline))
        @unroll_loop for i in 1:$(size(s, 1))
            @unroll_loop for j in 1:$(size(s, 2))
                @inbounds s[i, j] = x
            end
        end
    end
end
fast_fill!(s::T, x) where {T<:Array} = fill!(s, x)


# WARNING: HUGE PERFORMANCE HOTSPOT
function transform_bvn_derivs_hessian!(
    bvn_derivs::BivariateNormalDerivatives{T},
    sig_sf::GalaxySigmaDerivs{T},
    wcs_jacobian) where {T<:Number}

    @aliasscope begin
        # Hessian calculations.

        bvn_ss_h = bvn_derivs.bvn_ss_h
        bvn_us_h = bvn_derivs.bvn_us_h
        bvn_sig_d = Const(bvn_derivs.bvn_sig_d)
        #wcs_jacobian = Const(wcs_jacobian)
        sig_sf_j = sig_sf.j

        # Manually inlined version of fill!
        fast_fill!(bvn_ss_h, 0.0)
        fast_fill!(bvn_us_h, 0.0)

        # Second derviatives involving only shape parameters.
        # TODO: time consuming **************
        sig_sf_t = sig_sf.t
        @unroll_loop for shape_id2 in 1:GAL_SHAPE_IDS_LENGTH
            @unroll_loop for shape_id1 in 1:shape_id2
                @inbounds @unroll_loop for sig_id1 in 1:3
                    bvn_ss_h[shape_id1, shape_id2] += bvn_sig_d[sig_id1] * sig_sf_t[sig_id1, shape_id1, shape_id2]
                end
            end
        end

        bvn_sigsig_h = Const(bvn_derivs.bvn_sigsig_h)
        @unroll_loop for sig_id1 in 1:3
            @unroll_loop for sig_id2 in 1:3
                @inbounds @unroll_loop for shape_id2 in 1:GAL_SHAPE_IDS_LENGTH
                    inner_term = bvn_sigsig_h[sig_id1, sig_id2] * sig_sf_j[sig_id2, shape_id2]
                    @unroll_loop for shape_id1 in 1:shape_id2
                        bvn_ss_h[shape_id1, shape_id2] += inner_term * sig_sf_j[sig_id1, shape_id1]
                    end
                end
            end
        end

        @unroll_loop for shape_id2 in 1:GAL_SHAPE_IDS_LENGTH
            @inbounds @unroll_loop for shape_id1 in 1:shape_id2
                bvn_ss_h[shape_id2, shape_id1] = bvn_ss_h[shape_id1, shape_id2]
            end
        end

        # Second derivates involving both a shape term and a pos term.
        # TODO: time consuming **************
        bvn_xsig_h = Const(bvn_derivs.bvn_xsig_h)
        @unroll_loop for shape_id in 1:GAL_SHAPE_IDS_LENGTH
            @unroll_loop for u_id in 1:2
                @unroll_loop for sig_id in 1:3
                    @inbounds @unroll_loop for x_id in 1:2
                        bvn_us_h[u_id, shape_id] +=
                            bvn_xsig_h[x_id, sig_id] * sig_sf_j[sig_id, shape_id] * (-wcs_jacobian[x_id, u_id])
                    end
                end
            end
        end
    end
end


"""
Transform all the bvn derivatives from x and sigma to pos and the model
parameters.

These values should already have been set using get_bvn_derivs!():
bvn_x_d
bvn_xx_h
bvn_sig_d
bvn_sigsig_h
bvn_xsig_h
"""
function transform_bvn_derivs!(
    bvn_derivs::BivariateNormalDerivatives{T},
    sig_sf::GalaxySigmaDerivs{T},
    wcs_jacobian,
    calculate_hessian::Bool) where {T<:Number}

    # Transform the pos derivates first.
    # bvn_x_d and bvn_xx_h should already have been set using get_bvn_derivs!()
    transform_bvn_ux_derivs!(bvn_derivs, wcs_jacobian, calculate_hessian)

    # Gradient calculations.

    @aliasscope begin
        # Use the chain rule for the shape derviatives.
        # TODO: time consuming **************
        @aliasscope begin
            bvn_s_d   = bvn_derivs.bvn_s_d
            bvn_sig_d = Const(bvn_derivs.bvn_sig_d)
            sig_sf_j = sig_sf.j

            fast_fill!(bvn_s_d, 0.0)
            @unroll_loop for shape_id in 1:GAL_SHAPE_IDS_LENGTH
                @inbounds @unroll_loop for sig_id in 1:3
                    bvn_s_d[shape_id] += bvn_sig_d[sig_id] * sig_sf_j[sig_id, shape_id]
                end
            end
        end

        if calculate_hessian
            transform_bvn_derivs_hessian!(bvn_derivs, sig_sf, wcs_jacobian)
        end
    end
end

end  # module
