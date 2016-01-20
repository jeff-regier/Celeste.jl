
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
    dsiginv_dsig::Matrix{NumType}

    BvnComponent{T1 <: Number, T2 <: Number, T3 <: Number}(
        the_mean::Vector{T1}, the_cov::Matrix{T2}, weight::T3;
        calculate_siginv_deriv::Bool=true) = begin

      NumType = promote_type(T1, T2, T3);
      the_det = the_cov[1,1] * the_cov[2,2] - the_cov[1,2] * the_cov[2,1]
      c = 1 ./ (the_det^.5 * 2pi)

      if calculate_siginv_deriv
        # Derivatives of Sigma^{-1} with repsect to sigma.  These are the second
        # derivatives of log|Sigma| with respect to sigma.
        # dsiginv_dsig[a, b] is the derivative of sig^{-1}[a] / d sig[b]
        dsiginv_dsig = zeros(3, 3)

        precision = the_cov^-1

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
        new{NumType}(the_mean, precision, c * weight, dsiginv_dsig)
      else
        new{NumType}(the_mean, the_cov^-1, c * weight, zeros(0, 0))
      end

    end

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

  z = 1 + 2
  z2 = 3 + x[1]
  z3 = 3 + bmc.the_mean[2]
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

@doc """
Calculate the value, gradient, and hessian of
  -0.5 * x' sigma^-1 x - 0.5 * log|sigma|
with respect to x and sigma.

Args:
  - elbo_vars: A data structure with pre-allocated intermediate variables.
  - bvn: A bivariate normal component to get derivatives for.
  - calculate_sigma_hessian: Whether to also calculate derivatives with
      respect to sigma.  If false, only calculate x derivatives.
""" ->
function get_bvn_derivs!{NumType <: Number}(
    elbo_vars::ElboIntermediateVariables{NumType},
    bvn::BvnComponent{NumType}, x::Vector{Float64},
    calculate_x_hess::Bool,
    calculate_sigma_hessian::Bool)

  py1, py2, f_pre = eval_bvn_pdf(bvn, x);
  get_bvn_derivs!(
    elbo_vars, py1, py2, f_pre, bvn, calculate_x_hess, calculate_sigma_hessian)
end


function eval_bvn_log_density{NumType <: Number}(
    bvn::BvnComponent{NumType}, x::Vector{Float64})
  # This is the function of which get_bvn_derivs!() returns the derivatives.
  # It is only used for testing.

  py1, py2, f_pre = eval_bvn_pdf(bvn, x);
  -0.5 * (
    (x[1] - bvn.the_mean[1]) * py1 + (x[2] - bvn.the_mean[2]) * py2 -
    log(bvn.precision[1, 1] * bvn.precision[2, 2] - bvn.precision[1, 2] ^ 2))
end


function get_bvn_derivs!{NumType <: Number}(
    elbo_vars::ElboIntermediateVariables{NumType},
    py1::NumType, py2::NumType, f_pre::NumType,
    bvn::BvnComponent{NumType},
    calculate_x_hess::Bool,
    calculate_sigma_hessian::Bool)

  # Gradient with respect to x.
  bvn_x_d = elbo_vars.bvn_x_d
  bvn_x_d[1] = -py1
  bvn_x_d[2] = -py2

  if calculate_x_hess
    bvn_xx_h = elbo_vars.bvn_xx_h
    # println("-------------")
    # println(bvn_xx_h)
    # println(bvn.precision)

    # Hessian terms involving only x
    bvn_xx_h[1, 1] = -bvn.precision[1, 1]
    bvn_xx_h[2, 2] = -bvn.precision[2, 2]
    bvn_xx_h[1, 2] = bvn_xx_h[2, 1] = -bvn.precision[1 ,2]
  end

  # The first term is the derivative of -0.5 * x' Sigma^{-1} x
  # The second term is the derivative of -0.5 * log|Sigma|
  bvn_sig_d = elbo_vars.bvn_sig_d
  bvn_sig_d[1] = 0.5 * py1 * py1 - 0.5 * bvn.precision[1, 1]
  bvn_sig_d[2] = py1 * py2             - bvn.precision[1, 2]
  bvn_sig_d[3] = 0.5 * py2 * py2 - 0.5 * bvn.precision[2, 2]

  if calculate_sigma_hessian

    # Hessian calculation for terms containing sigma.

    # Derivatives of py1 and py2 with respect to s11, s12, s22 in that order.
    # These are used for the hessian calculations.
    dpy1_dsig = elbo_vars.dpy1_dsig
    dpy1_dsig[1] = -py1 * bvn.precision[1,1]
    dpy1_dsig[2] = -py2 * bvn.precision[1,1] - py1 * bvn.precision[1,2]
    dpy1_dsig[3] = -py2 * bvn.precision[1,2]

    dpy2_dsig = elbo_vars.dpy2_dsig
    dpy2_dsig[1] = -py1 * bvn.precision[1,2]
    dpy2_dsig[2] = -py1 * bvn.precision[2,2] - py2 * bvn.precision[1,2]
    dpy2_dsig[3] = -py2 * bvn.precision[2,2]

    # Hessian terms involving only sigma
    bvn_sigsig_h = elbo_vars.bvn_sigsig_h
    for s_ind=1:3
      # println("=============")
      # println(bvn_sigsig_h)
      # println(bvn.dsiginv_dsig)
      # Differentiate with respect to s_ind second.
      bvn_sigsig_h[1, s_ind] = #bvn_sigsig_h[s_ind, 1] =
        py1 * dpy1_dsig[s_ind] - 0.5 * bvn.dsiginv_dsig[1, s_ind]

      # d log|sigma| / dsigma12 is twice lambda12.
      bvn_sigsig_h[2, s_ind] =
        py1 * dpy2_dsig[s_ind] + py2 * dpy1_dsig[s_ind] -
        bvn.dsiginv_dsig[2, s_ind]

      bvn_sigsig_h[3, s_ind] = #bvn_sigsig_h[s_ind, 3] =
        py2 * dpy2_dsig[s_ind] - 0.5 * bvn.dsiginv_dsig[3, s_ind]
    end

    # Hessian terms involving both x and sigma.
    # Note that dpyA / dxB = bvn.precision[A, B]
    bvn_xsig_h = elbo_vars.bvn_xsig_h
    for x_ind=1:2
      bvn_xsig_h[x_ind, 1] = py1 * bvn.precision[1, x_ind]
      bvn_xsig_h[x_ind, 2] =
        py1 * bvn.precision[2, x_ind] + py2 * bvn.precision[1, x_ind]
      bvn_xsig_h[x_ind, 3] = py2 * bvn.precision[2, x_ind]
    end
  end

  #v
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
    XiXi::Matrix{NumType}; calculate_tensor::Bool=true) = begin

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
end


GalaxyCacheComponent{NumType <: Number}(
    e_dev_dir::Float64, e_dev_i::NumType,
    gc::GalaxyComponent, pc::PsfComponent, u::Vector{NumType},
    e_axis::NumType, e_angle::NumType, e_scale::NumType,
    calculate_derivs::Bool, calculate_hessian::Bool) = begin

  XiXi = Util.get_bvn_cov(e_axis, e_angle, e_scale)
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
    sig_sf = GalaxySigmaDerivs(Array(NumType, 0, 0), Array(NumType, 0, 0, 0))
  end

  GalaxyCacheComponent(e_dev_dir, e_dev_i, bmc, sig_sf)
end


@doc """
Transform the bvn derivatives and hessians from (x) to the
galaxy parameters (u).
""" ->
function transform_bvn_derivs!{NumType <: Number}(
    elbo_vars::ElboIntermediateVariables{NumType},
    bmc::BvnComponent{NumType},
    wcs_jacobian::Array{Float64, 2})

  bvn_u_d = elbo_vars.bvn_u_d
  bvn_uu_h = elbo_vars.bvn_uu_h

  # These values should already have been set using get_bvn_derivs!()
  bvn_x_d = elbo_vars.bvn_x_d
  bvn_xx_h = elbo_vars.bvn_xx_h

  # Gradient calculations.

  # Note that dxA_duB = -wcs_jacobian[A, B].  (It is minus the jacobian
  # because the object position affects the bvn.the_mean term, which is
  # subtracted from the pixel location as defined in bvn_sf.d.)
  bvn_u_d[1] =
    -(bvn_x_d[1] * wcs_jacobian[1, 1] + bvn_x_d[2] * wcs_jacobian[2, 1])
  bvn_u_d[2] =
    -(bvn_x_d[1] * wcs_jacobian[1, 2] + bvn_x_d[2] * wcs_jacobian[2, 2])


  if elbo_vars.calculate_hessian
    # Hessian calculations.

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
    bvn_uu_h[2, 1] = bvn_uu_h[1, 2]
  end
end


@doc """
Transform the bvn derivatives and hessians from (x, sigma) to the
galaxy parameters (u, gal_shape_ids).

You must have already called get_bvn_derivs!() before calling this.
""" ->
function transform_bvn_derivs!{NumType <: Number}(
    elbo_vars::ElboIntermediateVariables{NumType},
    gcc::GalaxyCacheComponent{NumType},
    wcs_jacobian::Array{Float64, 2})

  bvn_s_d = elbo_vars.bvn_s_d
  bvn_ss_h = elbo_vars.bvn_ss_h
  bvn_us_h = elbo_vars.bvn_us_h

  # These values should already have been set using get_bvn_derivs!()
  bvn_x_d = elbo_vars.bvn_x_d
  bvn_xx_h = elbo_vars.bvn_xx_h
  bvn_sig_d = elbo_vars.bvn_sig_d
  bvn_sigsig_h = elbo_vars.bvn_sigsig_h
  bvn_xsig_h = elbo_vars.bvn_xsig_h

  # Transform the u derivates first.
  transform_bvn_derivs!(elbo_vars, gcc.bmc, wcs_jacobian)

  # Gradient calculations.

  # Use the chain rule for the shape derviatives.
  # TODO: time consuming **************
  fill!(bvn_s_d, 0.0)
  @inbounds for shape_id in 1:length(gal_shape_ids), sig_id in 1:3
    bvn_s_d[shape_id] += bvn_sig_d[sig_id] * gcc.sig_sf.j[sig_id, shape_id]
  end

  if elbo_vars.calculate_hessian
    # Hessian calculations.

    fill!(bvn_ss_h, 0.0)
    fill!(bvn_us_h, 0.0)

    # Second derviatives involving only shape parameters.
    # TODO: time consuming **************
    @inbounds for shape_id2 in 1:length(gal_shape_ids), shape_id1 in 1:shape_id2
      @inbounds for sig_id1 in 1:3
        bvn_ss_h[shape_id1, shape_id2] +=
          bvn_sig_d[sig_id1] * gcc.sig_sf.t[sig_id1, shape_id1, shape_id2]
      end
    end

    @inbounds for sig_id1 in 1:3, sig_id2 in 1:3, shape_id2 in 1:length(gal_shape_ids)
      inner_term = bvn_sigsig_h[sig_id1, sig_id2] * gcc.sig_sf.j[sig_id2, shape_id2]
      @inbounds for shape_id1 in 1:shape_id2
        bvn_ss_h[shape_id1, shape_id2] +=
          inner_term * gcc.sig_sf.j[sig_id1, shape_id1]
      end
    end

    @inbounds for shape_id2 in 1:length(gal_shape_ids), shape_id1 in 1:shape_id2
      bvn_ss_h[shape_id2, shape_id1] = bvn_ss_h[shape_id1, shape_id2]
    end

    # Second derivates involving both a shape term and a u term.
    # TODO: time consuming **************
    @inbounds for shape_id in 1:length(gal_shape_ids), u_id in 1:2, sig_id in 1:3, x_id in 1:2
      bvn_us_h[u_id, shape_id] +=
        bvn_xsig_h[x_id, sig_id] * gcc.sig_sf.j[sig_id, shape_id] *
        (-wcs_jacobian[x_id, u_id])
    end
  end
end


@doc """
Convolve the current locations and galaxy shapes with the PSF.  If
calculate_derivs is true, also calculate derivatives and hessians for
active sources.

Args:
 - psf: A vector of PSF components
 - mp: The current ModelParams
 - b: The current band
 - calculate_derivs: Whether to calculate derivatives for active sources.

Returns:
 - star_mcs: An array of BvnComponents with indices
    - PSF component
    - Source (index within active_sources)
 - gal_mcs: An array of BvnComponents with indices
    - PSF component
    - Galaxy component
    - Galaxy type
    - Source (index within active_sources)
  Hessians are only populated for s in mp.active_sources.

The PSF contains three components, so you see lots of 3's below.
""" ->
function load_bvn_mixtures{NumType <: Number}(
    mp::ModelParams{NumType}, b::Int64;
    calculate_derivs::Bool=true, calculate_hessian::Bool=true)

  star_mcs = Array(BvnComponent{NumType}, 3, mp.S)
  gal_mcs = Array(GalaxyCacheComponent{NumType}, 3, 8, 2, mp.S)

  # TODO: do not keep any derviative information if the sources are not in
  # active_sources.
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
              for k = 1:3
                  gal_mcs[k, j, i, s] = GalaxyCacheComponent(
                      e_dev_dir, e_dev_i, galaxy_prototypes[i][j], psf[k],
                      m_pos, vs[ids.e_axis], vs[ids.e_angle], vs[ids.e_scale],
                      calculate_derivs && (s in mp.active_sources),
                      calculate_hessian)
              end
          end
      end
  end

  star_mcs, gal_mcs
end
