# Copied from SloanDigitalSkySurvey.PSF for more rapid iteration.
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
