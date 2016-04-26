module PSF

using Celeste
using Celeste.Model
using Celeste.BivariateNormals
import Celeste.SensitiveFloats

using Celeste.Transform
using Celeste.SensitiveFloats.SensitiveFloat
using Celeste.SensitiveFloats.clear!

import Optim
import WCS


 # Only include until this is merged with Optim.jl.
include("newton_trust_region.jl")


const ID_MAT_2D = eye(Float64, 2)

"""
A data type to store functions related to optimizing a PSF fit.  Initialize
using the transform and number of components, and then call fit_psf
to perform a fit to a specificed psf and initial parameters.
"""
type PsfOptimizer
  psf_transform::DataTransform
  ftol::Float64
  grtol::Float64
  num_iters::Int
  raw_psf::Matrix{Float64}
  K::Int

  # Variable that will be allocated in optimization:
  x_mat::Matrix{Float64}
  bvn_derivs::BivariateNormalDerivatives{Float64};

  log_pdf::SensitiveFloat{PsfParams, Float64};
  pdf::SensitiveFloat{PsfParams, Float64};
  pixel_value::SensitiveFloat{PsfParams, Float64};
  squared_error::SensitiveFloat{PsfParams, Float64};
  sf_free::SensitiveFloat{PsfParams, Float64};

  psf_params_free_vec_cache::Vector{Float64}

  # functions
  psf_2df::Optim.TwiceDifferentiableFunction
  fit_psf::Function

  function PsfOptimizer(psf_transform::DataTransform, K::Int;
                        verbose::Bool=false, ftol::Float64=1e-9, grtol::Float64=1e-9)

    num_iters = 50

    bvn_derivs = BivariateNormalDerivatives{Float64}(Float64);

    log_pdf = SensitiveFloats.zero_sensitive_float(PsfParams, Float64, 1);
    pdf = SensitiveFloats.zero_sensitive_float(PsfParams, Float64, 1);
    pixel_value = SensitiveFloats.zero_sensitive_float(PsfParams, Float64, K);
    squared_error = SensitiveFloats.zero_sensitive_float(PsfParams, Float64, K);
    sf_free = SensitiveFloats.zero_sensitive_float(PsfParams, Float64, K);

    x_mat = Array(Float64, 0, 0)
    raw_psf = Array(Float64, 0, 0)
    psf_params_free_vec_cache = fill(NaN, K * length(PsfParams))

    function psf_fit_for_optim{NumType <: Number}(
        psf_params_free_vec::Vector{NumType})

      if psf_params_free_vec == psf_params_free_vec_cache
        return sf_free
      else
        psf_params_free_vec_cache = deepcopy(psf_params_free_vec)
      end
      psf_params_free = unwrap_psf_params(psf_params_free_vec)
      psf_params = constrain_psf_params(psf_params_free, psf_transform)

      # Update squared_error in place.
      evaluate_psf_fit!(
          psf_params, raw_psf, x_mat, bvn_derivs,
          log_pdf, pdf, pixel_value, squared_error, true)

      # Update sf_free in place.
      transform_psf_sensitive_float!(
        psf_params, psf_transform, squared_error, sf_free, true)

      sf_free
    end

    function psf_fit_value{NumType <: Number}(psf_params_free_vec::Vector{NumType})
      psf_fit_for_optim(psf_params_free_vec).v[1]
    end

    function psf_fit_grad!(
        psf_params_free_vec::Vector{Float64}, grad::Vector{Float64})
      grad[:] = psf_fit_for_optim(psf_params_free_vec).d[:]
    end

    function psf_fit_hess!(
        psf_params_free_vec::Vector{Float64}, hess::Matrix{Float64})
      hess[:] = psf_fit_for_optim(psf_params_free_vec).h
      hess[:] = 0.5 * (hess + hess')
    end

    psf_2df = Optim.TwiceDifferentiableFunction(
      psf_fit_value, psf_fit_grad!, psf_fit_hess!)

    function fit_psf(psf::Matrix{Float64}, initial_params::Vector{Vector{Float64}})
      raw_psf = psf
      x_mat = get_x_matrix_from_psf(raw_psf);
      psf_params_free = unconstrain_psf_params(initial_params, psf_transform);
      psf_params_free_vec = vec(wrap_psf_params(psf_params_free));
      nm_result = newton_tr(psf_2df,
                            psf_params_free_vec,
                            xtol = 0.0, # Don't allow convergence in params
                            grtol = grtol,
                            ftol = ftol,
                            iterations = num_iters,
                            store_trace = verbose,
                            show_trace = verbose,
                            extended_trace = verbose,
                            initial_delta=10.0,
                            delta_hat=1e9,
                            rho_lower = 0.2)
      nm_result
    end

    new(psf_transform, ftol, grtol, num_iters, raw_psf, K,
      x_mat, bvn_derivs, log_pdf, pdf, pixel_value, squared_error, sf_free,
      psf_params_free_vec_cache, psf_2df, fit_psf)
  end
end


"""
Return an image of a Celeste GMM PSF evaluated at rows, cols.

Args:
- psf_array: The PSF to be evaluated as an array of PsfComponent
- rows: The rows in the image (usually in pixel coordinates)
- cols: The column in the image (usually in pixel coordinates)

Returns:
 - The PSF values at rows and cols.  The default size is the same as
   that returned by get_psf_at_point applied to FITS header values.

Note that the point in the image at which the PSF is evaluated --
that is, the center of the image returned by this function -- is
already implicit in the value of psf_array.
"""
function get_psf_at_point(psf_array::Array{PsfComponent, 1};
                         rows=collect(-25:25), cols=collect(-25:25))

  function get_psf_value(psf::PsfComponent, row::Float64, col::Float64)
     x = Float64[row, col] - psf.xiBar
     exp_term = exp(-0.5 * x' * psf.tauBarInv * x - 0.5 * psf.tauBarLd)
     (psf.alphaBar * exp_term / (2 * pi))[1]
  end

  Float64[
   sum([ get_psf_value(psf, float(row), float(col)) for psf in psf_array ])
     for row in rows, col in cols ]
end


"""
Get the PSF located at a particular world location in an image.

Args:
 - world_loc: A location in world coordinates.
 - img: An TiledImage

Returns:
 - An array of PsfComponent objects that represents the PSF as a mixture
   of Gaussians.
"""
function get_source_psf(world_loc::Vector{Float64}, img::TiledImage)
  # Some stamps or simulated data have no raw psf information.  In that case,
  # just use the psf from the image.
  if size(img.raw_psf_comp.rrows) == (0, 0)
    # Also return a vector of empty psf params
    return img.psf, fill(fill(NaN, length(PsfParams)), psf_K)
  else
    pixel_loc = WCS.world_to_pix(img.wcs, world_loc)
    psfstamp = img.raw_psf_comp(pixel_loc[1], pixel_loc[2])
    return PSF.fit_raw_psf_for_celeste(psfstamp)
  end
end


"""
Get the PSF located at a particular world location in an image.

Args:
 - world_loc: A location in world coordinates.
 - img: An TiledImage

Returns:
 - An array of PsfComponent objects that represents the PSF as a mixture
   of Gaussians.
"""
function get_source_psf(
    world_loc::Vector{Float64}, img::TiledImage,
    psf_optimizer::PSF.PsfOptimizer, initial_psf_params::Vector{Vector{Float64}})

  # Some stamps or simulated data have no raw psf information.  In that case,
  # just use the psf from the image.
  if size(img.raw_psf_comp.rrows) == (0, 0)
    return img.psf
  else
    pixel_loc = WCS.world_to_pix(img.wcs, world_loc)
    psfstamp = img.raw_psf_comp(pixel_loc[1], pixel_loc[2])
    return PSF.fit_raw_psf_for_celeste(psfstamp, psf_optimizer, initial_psf_params)
  end
end


"""
Get initial values for a set of PSF parameters.

Args:
  - K: The number of components.
  - for_test: If true, return a set of slightly asymmetric parameters so that
              derivatives are interesting for testing.

Returns:
  - A vector of PSF parameters, one for each component.
"""
function initialize_psf_params(K::Int; for_test::Bool=false)
  psf_params = Array(Vector{Float64}, K)
  for k=1:K
    if for_test
      # Choose asymmetric values for testing.
      psf_params[k] = zeros(length(PsfParams))
      psf_params[k][psf_ids.mu] = [0.1, 0.2]
      psf_params[k][psf_ids.e_axis] = 0.8
      psf_params[k][psf_ids.e_angle] = pi / 4
      psf_params[k][psf_ids.e_scale] = sqrt(2 * k)
      psf_params[k][psf_ids.weight] = 1 / K + k / 10
    else
      psf_params[k] = zeros(length(PsfParams))
      psf_params[k][psf_ids.mu] = [0.0, 0.0]
      psf_params[k][psf_ids.e_axis] = 0.95
      psf_params[k][psf_ids.e_angle] = 0.0
      psf_params[k][psf_ids.e_scale] = sqrt(2 * k)
      psf_params[k][psf_ids.weight] = 1 / K
    end
  end

  psf_params
end


"""
Given PSF parameters, return a DataTransform to transfrom them to and from
an unconstrained parameterization.

Args:
  - psf_params: The parameters to be transformed
  - scale: A vector as long as PsfParams to define a linear rescaling.

Returns:
  - A DataTransform object.
"""
function get_psf_transform(
    psf_params::Vector{Vector{Float64}};
    scale::Vector{Float64}=ones(length(PsfParams)))

  K = length(psf_params)
  bounds = Array(ParamBounds, length(psf_params))
  # Note that, for numerical reasons, the bounds must be on the scale
  # of reasonably meaningful changes.
  for k in 1:K
    bounds[k] = ParamBounds()
    bounds[k][:mu] = ParamBox[ ParamBox(-5.0, 5.0, scale[psf_ids.mu[1]]),
                               ParamBox(-5.0, 5.0, scale[psf_ids.mu[2]]) ]
    bounds[k][:e_axis] = ParamBox[ ParamBox(0.1, 1.0, scale[psf_ids.e_axis] ) ]
    bounds[k][:e_angle] =
      ParamBox[ ParamBox(-4 * pi, 4 * pi, scale[psf_ids.e_angle] ) ]
    bounds[k][:e_scale] =
      ParamBox[ ParamBox(0.05, 10.0, scale[psf_ids.e_scale] ) ]

    # Note that the weights do not need to sum to one.
    bounds[k][:weight] = ParamBox[ ParamBox(0.05, 2.0, scale[psf_ids.weight] ) ]
  end
  DataTransform(bounds, active_sources=collect(1:K), S=K)
end


"""
Transform psf_params to and from psf_params_free in place.

Args:
  - psf_params, psf_params_free: The constrained and unconstrained parameters,
      respectively.
  - psf_transform: The DataTransform to be applied.
  - to_unconstrained: If true, then update psf_params_free with the unconstrained
      representation of psf_params.  If false, update psf_params with the
      constrained version of psf_params_free.

Returns:
  - Updates psf_params or psf_params_free in place.
"""
function transform_psf_params!{NumType <: Number}(
    psf_params::Vector{Vector{NumType}}, psf_params_free::Vector{Vector{NumType}},
    psf_transform::DataTransform, to_unconstrained::Bool)

  for k=1:length(psf_params)
    for (param, constraint_vec) in psf_transform.bounds[k]
      for ind in 1:length(psf_ids.(param))
        param_ind = psf_ids.(param)[ind]
        constraint = constraint_vec[ind]
        to_unconstrained ?
          psf_params_free[k][param_ind] =
            Transform.unbox_parameter(psf_params[k][param_ind], constraint):
          psf_params[k][param_ind] =
            Transform.box_parameter(psf_params_free[k][param_ind], constraint)
      end
    end
  end

  true # return type
end


"""
Allocate memory for and return a constrained parameter set.
"""
function constrain_psf_params{NumType <: Number}(
    psf_params_free::Vector{Vector{NumType}}, psf_transform::DataTransform)

  K = length(psf_params_free)
  psf_params = Array(Vector{NumType}, K)
  for k=1:K
    psf_params[k] = zeros(NumType, length(PsfParams))
  end

  transform_psf_params!(psf_params, psf_params_free, psf_transform, false)

  psf_params
end


"""
Allocate memory for and return an unconstrained parameter set.
"""
function unconstrain_psf_params{NumType <: Number}(
    psf_params::Vector{Vector{NumType}}, psf_transform::DataTransform)

  K = length(psf_params)
  psf_params_free = Array(Vector{NumType}, K)
  for k=1:K
    psf_params_free[k] = zeros(NumType, length(PsfParams))
  end

  transform_psf_params!(psf_params, psf_params_free, psf_transform, true)

  psf_params_free
end


"""
Convert a single vector of psf parameters to a vector of vectors.
"""
function unwrap_psf_params{NumType <: Number}(psf_param_vec::Vector{NumType})
  @assert length(psf_param_vec) % length(PsfParams) == 0
  K = round(Int, length(psf_param_vec) / length(PsfParams))
  psf_param_mat = reshape(psf_param_vec, length(PsfParams), K)
  psf_params = Array(Vector{NumType}, K)
  for k = 1:K
    psf_params[k] = psf_param_mat[:, k]
  end
  psf_params
end


"""
Convert a vector of vectors of psf parameters to a single vector.
"""
function wrap_psf_params{NumType <: Number}(psf_params::Vector{Vector{NumType}})
  psf_params_mat = zeros(NumType, length(PsfParams), length(psf_params))
  for k=1:length(psf_params)
    psf_params_mat[:, k] = psf_params[k]
  end
  psf_params_mat[:]
end


"""
Return a sensitive float representing the value of the psf at pixel x
with all its associated derivatives (with respect to the constrained
parameterization).  Note that "fit" is a bad name -- it is just the pixel value.
TODO: fix the name.

Args:
  - x: The 2d location at which to evaluate the pdf
  - calculate_derivs: If false, only update the value of pixel_value
  - other values: Pre-allocated memory for intermediate calculations

Returns:
  - Updates pixel_value in place (and all the other placeholder values as well)
"""
function evaluate_psf_pixel_fit!{NumType <: Number}(
    x::Vector{Float64}, psf_params::Vector{Vector{NumType}},
    sigma_vec::Vector{Matrix{NumType}},
    sig_sf_vec::Vector{GalaxySigmaDerivs{NumType}},
    bvn_vec::Vector{BvnComponent{NumType}},
    bvn_derivs::BivariateNormalDerivatives{NumType},
    log_pdf::SensitiveFloat{PsfParams, NumType},
    pdf::SensitiveFloat{PsfParams, NumType},
    pixel_value::SensitiveFloat{PsfParams, NumType},
    calculate_derivs::Bool)

  clear!(pixel_value)

  K = length(psf_params)
  sigma_ids = (psf_ids.e_axis, psf_ids.e_angle, psf_ids.e_scale)
  for k = 1:K
    # I will put in the weights later so that the log pdf sensitive float
    # is accurate.
    bvn = bvn_vec[k];
    eval_bvn_pdf!(bvn_derivs, bvn, x)
    get_bvn_derivs!(bvn_derivs, bvn, true, true)
    transform_bvn_derivs!(bvn_derivs, sig_sf_vec[k], ID_MAT_2D, true)

    clear!(log_pdf)
    clear!(pdf)

    # This is redundant, but it's what eval_bvn_pdf returns.
    log_pdf.v[1] = log(bvn_derivs.f_pre[1])

    if calculate_derivs
      for ind=1:2
        log_pdf.d[psf_ids.mu[ind]] = bvn_derivs.bvn_u_d[ind]
      end
      for ind=1:3
        log_pdf.d[sigma_ids[ind]] = bvn_derivs.bvn_s_d[ind]
      end
      log_pdf.d[psf_ids.weight] = 0

      for ind1 = 1:2, ind2 = 1:2
        log_pdf.h[psf_ids.mu[ind1], psf_ids.mu[ind2]] =
          bvn_derivs.bvn_uu_h[ind1, ind2]
      end
      for mu_ind = 1:2, sig_ind = 1:3
        log_pdf.h[psf_ids.mu[mu_ind], sigma_ids[sig_ind]] =
        log_pdf.h[sigma_ids[sig_ind], psf_ids.mu[mu_ind]] =
          bvn_derivs.bvn_us_h[mu_ind, sig_ind]
      end
      for ind1 = 1:3, ind2 = 1:3
        log_pdf.h[sigma_ids[ind1], sigma_ids[ind2]] =
          bvn_derivs.bvn_ss_h[ind1, ind2]
      end
    end

    pdf_val = exp(log_pdf.v[1])
    pdf.v[1] = pdf_val

    if calculate_derivs
      for ind1 = 1:length(PsfParams)
        if ind1 == psf_ids.weight
          pdf.d[ind1] = pdf_val
        else
          pdf.d[ind1] = psf_params[k][psf_ids.weight] * pdf_val * log_pdf.d[ind1]
        end

        for ind2 = 1:ind1
          pdf.h[ind1, ind2] = pdf.h[ind2, ind1] =
            psf_params[k][psf_ids.weight] * pdf_val *
            (log_pdf.h[ind1, ind2] + log_pdf.d[ind1] * log_pdf.d[ind2])
        end
      end

      # Weight hessian terms.
      for ind1 = 1:length(PsfParams)
        pdf.h[psf_ids.weight, ind1] = pdf.h[ind1, psf_ids.weight] =
          pdf_val * log_pdf.d[ind1]
      end

    end

    pdf.v[1] *= psf_params[k][psf_ids.weight]

    SensitiveFloats.add_sources_sf!(pixel_value, pdf, k, calculate_derivs)
  end

  true # Set return type
end


"""
Convert PSF parameters to covariance matrices and derivatives and BvnComponents.
"""
function get_sigma_from_params{NumType <: Number}(
    psf_params::Vector{Vector{NumType}})

  K = length(psf_params)
  sigma_vec = Array(Matrix{NumType}, K);
  sig_sf_vec = Array(GalaxySigmaDerivs{NumType}, K);
  bvn_vec = Array(BvnComponent{NumType}, K);
  for k = 1:K
    sigma_vec[k] = BivariateNormals.get_bvn_cov(psf_params[k][psf_ids.e_axis],
                                    psf_params[k][psf_ids.e_angle],
                                    psf_params[k][psf_ids.e_scale])
    sig_sf_vec[k] = GalaxySigmaDerivs(
      psf_params[k][psf_ids.e_angle],
      psf_params[k][psf_ids.e_axis],
      psf_params[k][psf_ids.e_scale], sigma_vec[k], calculate_tensor=true);

    bvn_vec[k] =
      BvnComponent{NumType}(psf_params[k][psf_ids.mu], sigma_vec[k], 1.0);
  end
  sigma_vec, sig_sf_vec, bvn_vec
end


"""
Evaluate the sum of squared difference between the raw_psf and the psf
represented by psf_params, as well as the derivatives and hessians with
respect to unconstrained parameters.

Returns:
  - A sensitive float for the sum of squared differences.
"""
function evaluate_psf_fit{NumType <: Number}(
    psf_params::Vector{Vector{NumType}}, raw_psf::Matrix{Float64},
    calculate_derivs::Bool)

  K = length(psf_params)
  x_mat = get_x_matrix_from_psf(raw_psf);

  # TODO: allocate these outside?
  bvn_derivs = BivariateNormalDerivatives{NumType}(NumType);
  log_pdf = SensitiveFloats.zero_sensitive_float(PsfParams, NumType, 1);
  pdf = SensitiveFloats.zero_sensitive_float(PsfParams, NumType, 1);

  pixel_value = SensitiveFloats.zero_sensitive_float(PsfParams, NumType, K);
  squared_error = SensitiveFloats.zero_sensitive_float(PsfParams, NumType, K);

  evaluate_psf_fit!(
      psf_params, raw_psf, x_mat, bvn_derivs,
      log_pdf, pdf, pixel_value, squared_error, calculate_derivs)

  squared_error
end


"""
evaluate_psf_fit but with pre-allocated memory for intermediate calculations.
"""
function evaluate_psf_fit!{NumType <: Number}(
    psf_params::Vector{Vector{NumType}}, raw_psf::Matrix{Float64},
    x_mat::Matrix{Vector{Float64}},
    bvn_derivs::BivariateNormalDerivatives{NumType},
    log_pdf::SensitiveFloat{PsfParams, NumType},
    pdf::SensitiveFloat{PsfParams, NumType},
    pixel_value::SensitiveFloat{PsfParams, NumType},
    squared_error::SensitiveFloat{PsfParams, NumType},
    calculate_derivs::Bool)

  K = length(psf_params)
  sigma_vec, sig_sf_vec, bvn_vec = get_sigma_from_params(psf_params)
  clear!(squared_error)

  for x_ind in 1:length(x_mat)
    clear!(pixel_value)
    evaluate_psf_pixel_fit!(
        x_mat[x_ind], psf_params, sigma_vec, sig_sf_vec, bvn_vec,
        bvn_derivs, log_pdf, pdf, pixel_value, calculate_derivs)

    diff = (pixel_value.v[1] - raw_psf[x_ind])
    squared_error.v[1] +=  diff ^ 2
    if calculate_derivs
      for ind1 = 1:length(squared_error.d)
        squared_error.d[ind1] += 2 * diff * pixel_value.d[ind1]
        for ind2 = 1:ind1
          squared_error.h[ind1, ind2] +=
            2 * (diff * pixel_value.h[ind1, ind2] +
                 pixel_value.d[ind1] * pixel_value.d[ind2]')
          squared_error.h[ind2, ind1] = squared_error.h[ind1, ind2]
        end
      end
    end # if calculate_derivs
  end

  squared_error
end


"""
Transform a sf, a SensitiveFloat that has derivatives with respect to
constrianed parameters, into sf_free, a SensitiveFloat that has derivatives
with respect to unconstrained parameteres.

Args:
  - psf_params: The constrianed parameters
  - psf_transform: The data transform to be applied
  - sf: The SensitiveFloat with derivatives with respect to the constrained params
  - sf_free: Updated in place.  The SensitiveFloat with derivatives with respect
             to the unconstrained parameters.
  - calculate_derivs: If false, only calculate the value.

Returns:
  - Updates sf_free in place.
"""
function transform_psf_sensitive_float!{NumType <: Number}(
    psf_params::Vector{Vector{NumType}}, psf_transform::Transform.DataTransform,
    sf::SensitiveFloat{PsfParams, NumType}, sf_free::SensitiveFloat{PsfParams, NumType},
    calculate_derivs::Bool)

  sf_free.v[1] = sf.v[1]
  if calculate_derivs
    K = length(psf_params)

    # These are the hessians of each individual parameter's transform.  We
    # can represent it this way since each parameter's transform only depends on
    # its own value and not on others.

    # This is the diagonal of the Jacobian transform.
    jacobian_diag = zeros(length(PsfParams) * K);

    # These are hte Hessians of the each parameter's transform.
    hessian_values = zeros(length(PsfParams) * K);

    for k = 1:K
      offset = length(PsfParams) * (k - 1)
      for ind = 1:2
        mu_ind = psf_ids.mu[ind]
        jac, hess = Transform.box_derivatives(
          psf_params[k][mu_ind], psf_transform.bounds[k][:mu][ind]);
        jacobian_diag[offset + mu_ind] = jac
        hessian_values[offset + mu_ind] = hess
      end

      # The rest are one-dimensional.
      for field in setdiff(fieldnames(PsfParams), [ :mu ])
        ind = psf_ids.(field)
        jac, hess = Transform.box_derivatives(
          psf_params[k][ind], psf_transform.bounds[k][field][1]);
        jacobian_diag[offset + ind] = jac
        hessian_values[offset + ind] = hess
      end
    end

    # Apply the transformations.
    for ind1 = 1:length(sf_free.d)
      sf_free.d[ind1] = jacobian_diag[ind1] * sf.d[ind1]

      for ind2 = 1:ind1
        # Calculate the Hessian
        sf_free.h[ind1, ind2] = sf_free.h[ind2, ind1] =
          (jacobian_diag[ind1] * jacobian_diag[ind2]) * sf.h[ind1, ind2]
        if ind1 == ind2
          sf_free.h[ind1, ind2] +=  hessian_values[ind1] * sf.d[ind1]
          diagm(hessian_values .* sf.d[:])
        end
      end
    end
  end

  true # return type
end


"""
Given a psf matrix, return a matrix of the same size, where each element of
the matrix is a 2-length vector of the [x1, x2] location of that matrix cell.
The locations are chosen so that the scale is pixels, but the center of the
psf is [0, 0].
"""
function get_x_matrix_from_psf(psf::Matrix{Float64})
  psf_center = Float64[ (size(psf, i) - 1) / 2 + 1 for i=1:2 ]
  Vector{Float64}[ Float64[i, j] - psf_center for i=1:size(psf, 1), j=1:size(psf, 2) ]
end


"""
Fit a mixture of 2d Gaussians to a PSF image (evaluated at a single point).

Args:
  - raw_psf: An matrix image of the point spread function.
  - psf_optimizer: A PsfOptimizer object for the fit.

Returns:
  - A vector of PsfComponents fit to the raw_psf.
"""
function fit_raw_psf_for_celeste(
    raw_psf::Array{Float64, 2}, psf_optimizer::PsfOptimizer,
    initial_psf_params::Vector{Vector{Float64}})

  K = length(initial_psf_params)
  @assert K == psf_optimizer.K
  optim_result = psf_optimizer.fit_psf(raw_psf, initial_psf_params)
  psf_params_fit =
    constrain_psf_params(
      unwrap_psf_params(optim_result.minimum), psf_optimizer.psf_transform)

  sigma_vec = get_sigma_from_params(psf_params_fit)[1]
  celeste_psf = Array(PsfComponent, K)
  for k=1:K
    mu = psf_params_fit[k][psf_ids.mu]
    weight = psf_params_fit[k][psf_ids.weight]
    celeste_psf[k] = PsfComponent(weight, mu, sigma_vec[k])
  end

  celeste_psf, psf_params_fit
end


"""
Fit a mixture of 2d Gaussians to a PSF image (evaluated at a single point).

Args:
 - raw_psf: An matrix image of the point spread function.
 - K: The number of components to fit.

 Returns:
   - A vector of PsfComponents fit to the raw_psf.
"""
function fit_raw_psf_for_celeste(raw_psf::Array{Float64, 2}; K=psf_K, ftol=1e-9)
  psf_params = PSF.initialize_psf_params(K, for_test=false);
  psf_transform = PSF.get_psf_transform(psf_params);
  psf_optimizer = PsfOptimizer(psf_transform, K);
  psf_optimizer.ftol = ftol
  fit_raw_psf_for_celeste(raw_psf, psf_optimizer, psf_params)
end

end # module
