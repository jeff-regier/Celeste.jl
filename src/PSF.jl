module PSF

using Celeste
using Celeste.Types
using Celeste.BivariateNormals
import Celeste.Util
import Celeste.SensitiveFloats
import Celeste.SDSSIO

using Celeste.Transform
#
# using Celeste.Transform.ParamBounds
# using Celeste.Transform.ParamBox
# using Celeste.Transform.DataTransform
# using Celeste.Transform.box_parameter
# using Celeste.Transform.unbox_parameter

using Celeste.SensitiveFloats.SensitiveFloat
using Celeste.SensitiveFloats.clear!

export evaluate_psf_fit, psf_params_to_array, psf_array_to_params,
       get_psf_transform, initialize_psf_params, transform_psf_params!,
       unwrap_psf_params, wrap_psf_params,
       unconstrain_psf_params, constrain_psf_params,
       transform_psf_sensitive_float!


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

function get_psf_transform(psf_params::Vector{Vector{Float64}})

  K = length(psf_params)
  bounds = Array(ParamBounds, length(psf_params))
  # Note that, for numerical reasons, the bounds must be on the scale
  # of reasonably meaningful changes.
  for k in 1:K
    bounds[k] = ParamBounds()
    bounds[k][:mu] = fill(ParamBox(-5.0, 5.0, 1.0), 2)
    bounds[k][:e_axis] = ParamBox[ ParamBox(0.1, 1.0, 100.0) ]
    bounds[k][:e_angle] = ParamBox[ ParamBox(-4 * pi, 4 * pi, 1.0) ]
    bounds[k][:e_scale] = ParamBox[ ParamBox(0.05, 10.0, 1.0) ]

    # Note that the weights do not need to sum to one.
    bounds[k][:weight] = ParamBox[ ParamBox(0.05, 2.0, 1.0) ]
  end
  DataTransform(bounds, active_sources=collect(1:K), S=K)
end


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


function wrap_psf_params{NumType <: Number}(psf_params::Vector{Vector{NumType}})
  psf_params_mat = zeros(NumType, length(PsfParams), length(psf_params))
  for k=1:length(psf_params)
    psf_params_mat[:, k] = psf_params[k]
  end
  psf_params_mat[:]
end

# function psf_params_to_vec{NumType <: Number}(psf_params::Vector{Vector{NumType}})
#   K = length(psf_params)
#   psf_params_mat = zeros(NumType, length(PsfParams), K)
#   for k=1:K
#     psf_params_mat[:, k] = psf_params[k]
#   end
#   psf_params_mat
# end
#
#
# function psf_vec_to_params{NumType <: Number}(psf_params_mat::Matrix{NumType})
#   K = size(psf_params_mat, 2)
#   @assert size(psf_params_mat, 1) == length(PsfParams)
#   psf_params = Array(Vector{NumType}, K)
#   for k=1:K
#     # psf_params[k] = zeros(NumType, length(PsfParams))
#     psf_params[k] = psf_params_mat[:, k]
#   end
# end
#

function evaluate_psf_pixel_fit!{NumType <: Number}(
    x::Vector{Float64}, psf_params::Vector{Vector{NumType}},
    sigma_vec::Vector{Matrix{NumType}},
    sig_sf_vec::Vector{GalaxySigmaDerivs{NumType}},
    bvn_derivs::BivariateNormalDerivatives{NumType},
    log_pdf::SensitiveFloat{PsfParams, NumType},
    pdf::SensitiveFloat{PsfParams, NumType},
    pixel_value::SensitiveFloat{PsfParams, NumType},
    calculate_derivs::Bool)

  clear!(pixel_value)

  K = length(psf_params)
  sigma_ids = [psf_ids.e_axis, psf_ids.e_angle, psf_ids.e_scale]
  for k = 1:K
    # I will put in the weights later so that the log pdf sensitive float
    # is accurate.
    bvn = BvnComponent{NumType}(psf_params[k][psf_ids.mu], sigma_vec[k], 1.0);
    eval_bvn_pdf!(bvn_derivs, bvn, x)
    get_bvn_derivs!(bvn_derivs, bvn, true, true)
    transform_bvn_derivs!(bvn_derivs, sig_sf_vec[k], eye(Float64, 2), true)

    clear!(log_pdf)
    clear!(pdf)

    # This is redundant, but it's what eval_bvn_pdf returns.
    log_pdf.v[1] = log(bvn_derivs.f_pre[1])

    if calculate_derivs
      log_pdf.d[psf_ids.mu] = bvn_derivs.bvn_u_d
      log_pdf.d[sigma_ids] = bvn_derivs.bvn_s_d
      log_pdf.d[psf_ids.weight] = 0

      log_pdf.h[psf_ids.mu, psf_ids.mu] = bvn_derivs.bvn_uu_h
      log_pdf.h[psf_ids.mu, sigma_ids] = bvn_derivs.bvn_us_h
      log_pdf.h[sigma_ids, psf_ids.mu] = log_pdf.h[psf_ids.mu, sigma_ids]'
      log_pdf.h[sigma_ids, sigma_ids] = bvn_derivs.bvn_ss_h
    end

    pdf_val = exp(log_pdf.v[1])
    pdf.v[1] = pdf_val

    if calculate_derivs
      pdf.d = pdf_val * log_pdf.d
      pdf.h = pdf_val * (log_pdf.h + log_pdf.d * log_pdf.d')

      # Now multiply by the weight.
      pdf.h *= psf_params[k][psf_ids.weight]
      pdf.h[psf_ids.weight, :] = pdf.h[:, psf_ids.weight] = pdf.d

      pdf.d *= psf_params[k][psf_ids.weight]
      pdf.d[psf_ids.weight] = pdf_val
    end

    pdf.v *= psf_params[k][psf_ids.weight]

    SensitiveFloats.add_sources_sf!(pixel_value, pdf, k, calculate_derivs)
  end

  true # Set return type
end


function get_sigma_from_params{NumType <: Number}(
    psf_params::Vector{Vector{NumType}})

  K = length(psf_params)
  sigma_vec = Array(Matrix{NumType}, K);
  sig_sf_vec = Array(GalaxySigmaDerivs{NumType}, K);
  for k = 1:K
    sigma_vec[k] = Util.get_bvn_cov(psf_params[k][psf_ids.e_axis],
                                    psf_params[k][psf_ids.e_angle],
                                    psf_params[k][psf_ids.e_scale])
    sig_sf_vec[k] = GalaxySigmaDerivs(
      psf_params[k][psf_ids.e_angle],
      psf_params[k][psf_ids.e_axis],
      psf_params[k][psf_ids.e_scale], sigma_vec[k], calculate_tensor=true);

  end
  sigma_vec, sig_sf_vec
end


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
  sigma_vec, sig_sf_vec = get_sigma_from_params(psf_params)

  clear!(squared_error)

  for x_ind in 1:length(x_mat)
    clear!(pixel_value)
    evaluate_psf_pixel_fit!(
        x_mat[x_ind], psf_params, sigma_vec, sig_sf_vec,
        bvn_derivs, log_pdf, pdf, pixel_value, calculate_derivs)

    diff = (pixel_value.v[1] - raw_psf[x_ind])
    squared_error.v +=  diff ^ 2
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
    end
  end

  squared_error
end


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
        sf_free.h[ind1, ind2] =
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

























#############################################
# Below was copied from SloanDigitalSkySurvey.PSF for more rapid iteration.
# TODO: remove this redundant code from one package or the other.

using Optim

# Constants to keep the optimization stable.
const sigma_min = diagm([0.25, 0.25])
const weight_min = 0.05


"""
Convert the parameters of a mixture of 2d Gaussians
sum_k weight_vec[k] * N(mu_vec[k], sigma_vec[k])
to an unconstrained vector that can be passed to an optimizer.
"""
function wrap_parameters(
    mu_vec::Vector{Vector{Float64}},
    sigma_vec::Vector{Matrix{Float64}},
    weight_vec::Vector{Float64})

  @assert length(mu_vec) == length(sigma_vec) == length(weight_vec)
  local K = length(mu_vec)

  # Two mean parameters, three covariance parameters,
  # and one weight per component.
  local par = zeros(K * 6)
  for k = 1:K
    offset = (k - 1) * 6
    par[offset + (1:2)] = mu_vec[k]

    sigma_chol = chol(sigma_vec[k] - sigma_min)
    par[offset + 3] = log(sigma_chol[1, 1])
    par[offset + 4] = sigma_chol[1, 2]
    par[offset + 5] = log(sigma_chol[2, 2])

    # An alternative parameterization:
    # sigma_diff = sigma_vec[k] - sigma_min
    # sigma_diff11 = sqrt(max(sigma_diff[1, 1], 0.0))
    # sigma_diff22 = sqrt(max(sigma_diff[2, 2], 0.0))
    # rho = sigma_diff[1, 2] / (sigma_diff11 * sigma_diff22)
    # if abs(rho) > 1
    #   rho = rho / abs(rho)
    # end
    # par[offset + 3] = log(sigma_diff11)
    # par[offset + 4] = atanh(rho)
    # par[offset + 5] = log(sigma_diff22)

    par[offset + 6] = log(weight_vec[k] - weight_min)
  end

  par
end


"""
Reverse wrap_parameters().
"""
function unwrap_parameters{T <: Number}(par::Vector{T})
  local K = round(Int, length(par) / 6)
  @assert K == length(par) / 6

  local mu_vec = Array(Vector{T}, K)
  local sigma_vec = Array(Matrix{T}, K)
  local weight_vec = zeros(T, K)

  for k = 1:K
    offset = (k - 1) * 6
    mu_vec[k] = par[offset + (1:2)]

    sigma_chol_vec = par[offset + (3:5)]
    sigma_chol = T[exp(sigma_chol_vec[1]) sigma_chol_vec[2];
                   0.0                    exp(sigma_chol_vec[3])]
    sigma_vec[k] = sigma_chol' * sigma_chol + sigma_min

    # An alternative parameterization:
    # sigma_diff11 = exp(par[offset + 3])
    # rho = tanh(par[offset + 4])
    # sigma_diff22 = exp(par[offset + 5])
    # sigma_vec[k] = T[ sigma_diff11 ^ 2 sigma_diff11 * sigma_diff22 * rho;
    #                   sigma_diff11 * sigma_diff22 * rho sigma_diff22 ^ 2] +
    #               sigma_min

    weight_vec[k] = exp(par[offset + 6]) + weight_min
  end

  mu_vec, sigma_vec, weight_vec
end


"""
Using a vector of mu, sigma, and weights, get the value of the GMM at x.

Args:
  - x: A length 2 vector to evaluate the GMM at.
  - mu_vec: A vector of location vectors.
  - sigma_vec: A vector of covariance matrices.
  - weight_vec: A vector of weights.

Returns:
  - The weighted sum of the component densities at x.
"""
function evaluate_psf_at_point{T <: Number}(
    x::Vector{Float64},
    mu_vec::Vector{Vector{T}},
    sigma_vec::Vector{Matrix{T}},
    weight_vec::Vector{T})

  @assert length(mu_vec) == length(sigma_vec) == length(weight_vec)
  local K = length(mu_vec)

  @assert length(x) == 2
  local pdf = 0.0
  for k = 1:K
    z = x - mu_vec[k]
    log_pdf = -0.5 * dot(z, sigma_vec[k] \ z) -
               0.5 * logdet(sigma_vec[k]) - log(2 * pi)
    pdf += weight_vec[k] * exp(log_pdf)
  end

  pdf
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
Evaluate a GMM expressed in its unconstrained form.
"""
function render_psf{T <: Number}(par::Vector{T}, x_mat::Matrix{Vector{Float64}})
  local mu_vec, sigma_vec, weight_vec
  mu_vec, sigma_vec, weight_vec = unwrap_parameters(par)

  gmm_psf =
    T[ evaluate_psf_at_point(x_mat[i, j], mu_vec, sigma_vec, weight_vec)
       for i=1:size(x_mat, 1), j=1:size(x_mat, 2) ]
  gmm_psf
end


"""
Fit a mixture of 2d Gaussians to a PSF image (evaluated at a single point).

Args:
 - psf: An matrix image of the point spread function, e.g. as returned by
        get_psf_at_point.

"""
function fit_psf_gaussians_least_squares(
    psf::Array{Float64, 2}; initial_par=Float64[],
    ftol = 1e-5, iterations = 5000, verbose=false,
    K=2, optim_method=:nelder_mead)

  println("Using Celeste version of psf fit.")

  if (any(psf .< 0))
      if verbose
          warn("Some psf values are negative.")
      end
      psf[ psf .< 0 ] = 0
  end

  x_mat = get_x_matrix_from_psf(psf);

  if length(initial_par) == 0
    psf_starting_mean  =
      sum([ psf[i, j]  * x_mat[i, j] for
          i in 1:size(psf, 1), j=1:size(psf, 2) ]) / sum(psf)

    mu_vec = Array(Vector{Float64}, K)
    sigma_vec = Array(Matrix{Float64}, K)
    weight_vec = zeros(Float64, K)

    for k=1:K
      mu_vec[k] = psf_starting_mean
      sigma_vec[k] = Float64[ sqrt(2 * k) 0; 0 sqrt(2 * k)]
      weight_vec[k] = 1 / K
    end
    if verbose
      println("Using default initialization:")
    end
    initial_par = wrap_parameters(mu_vec, sigma_vec, weight_vec)
  end

  @assert length(initial_par) == K * 6

  function evaluate_fit{T <: Number}(par::Vector{T})
    gmm_psf = render_psf(par, x_mat)
    local fit = sum((psf .- gmm_psf) .^ 2)
    if verbose && T == Float64
      println("-------------------")
      println("Fit: $fit")
      mu_vec, sigma_vec, weight_vec = unwrap_parameters(par)
    end
    fit
  end

  optim_result =
    Optim.optimize(evaluate_fit, initial_par, method=optim_method,
                   iterations=iterations, ftol=ftol)

  optim_result, unwrap_parameters(optim_result.minimum)...
end

end # module
