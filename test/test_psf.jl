using Celeste
using Celeste.Types
using Celeste.SkyImages
using Celeste.BivariateNormals
using  Celeste.PSF

import Celeste.Util
import Celeste.SensitiveFloats
import Celeste.SDSSIO

using ForwardDiff

using Base.Test

function test_psf_fit()
  run_num = 4263
  camcol_num = 5
  field_num = 117
  b = 3

  psf_filename =
    @sprintf("%s/psField-%06d-%d-%04d.fit", datadir, run_num, camcol_num, field_num)
  psf_fits = FITSIO.FITS(psf_filename);
  raw_psf_comp = SDSSIO.read_psf(psf_fits, band_letters[b]);
  close(psf_fits)

  raw_psf = raw_psf_comp(500., 500.);

  # Initialize params
  K = 2
  psf_params = zeros(length(Types.PsfParams), K)
  for k=1:K
    psf_params[psf_ids.mu, k] = [0., 0.]
    psf_params[psf_ids.e_axis, k] = 0.8
    psf_params[psf_ids.e_angle, k] = pi / 4
    psf_params[psf_ids.e_scale, k] = sqrt(2 * k)
    psf_params[psf_ids.weight, k] = 1/ K
  end

  function pixel_value_wrapper_sf{NumType <: Number}(
      psf_param_vec::Vector{NumType}, calculate_derivs::Bool)

    local psf_params = reshape(psf_param_vec, length(PsfParams), 2)

    bvn_derivs = BivariateNormalDerivatives{NumType}(NumType);
    log_pdf = SensitiveFloats.zero_sensitive_float(PsfParams, NumType, 1);
    pdf = SensitiveFloats.zero_sensitive_float(PsfParams, NumType, 1);

    local pixel_value = SensitiveFloats.zero_sensitive_float(PsfParams, NumType, K);

    sigma_vec = Array(Matrix{NumType}, K);
    sig_sf_vec = Array(GalaxySigmaDerivs{NumType}, K);

    for k = 1:K
      sigma_vec[k] = Util.get_bvn_cov(psf_params[psf_ids.e_axis, k],
                                      psf_params[psf_ids.e_angle, k],
                                      psf_params[psf_ids.e_scale, k])
      sig_sf_vec[k] = GalaxySigmaDerivs(
        psf_params[psf_ids.e_angle, k],
        psf_params[psf_ids.e_axis, k],
        psf_params[psf_ids.e_scale, k], sigma_vec[k], calculate_tensor=calculate_derivs);

    end

    SensitiveFloats.clear!(pixel_value)
    PSF.evaluate_psf_pixel_fit!(
        x, psf_params, sigma_vec, sig_sf_vec,
        bvn_derivs, log_pdf, pdf, pixel_value, calculate_derivs)

    pixel_value
  end

  function pixel_value_wrapper_value{NumType <: Number}(psf_param_vec::Vector{NumType})
    pixel_value_wrapper_sf(psf_param_vec, false).v[1]
  end


  # Pick a point for testing
  # x_ind = 1508
  # x_mat = PSF.get_x_matrix_from_psf(raw_psf);
  # x = x_mat[x_ind]
  x = Float64[1.0, 2.0]

  sigma_vec = Array(Matrix{Float64}, K);
  for k = 1:K
    sigma_vec[k] = Util.get_bvn_cov(psf_params[psf_ids.e_axis, k],
                                    psf_params[psf_ids.e_angle, k],
                                    psf_params[psf_ids.e_scale, k])
  end

  println("Testing single pixel value")
  psf_components = PsfComponent[
    PsfComponent(psf_params[psf_ids.weight, k], psf_params[psf_ids.mu, k], sigma_vec[k])
                  for k = 1:K ];

  psf_rendered = get_psf_at_point(psf_components, rows=[ x[1] ], cols=[ x[2] ])[1];
  @test_approx_eq psf_rendered pixel_value_wrapper_value(psf_params[:])

  pixel_value = deepcopy(pixel_value_wrapper_sf(psf_params[:], true));

  ad_grad = ForwardDiff.gradient(pixel_value_wrapper_value, psf_params[:]);
  ad_hess = ForwardDiff.hessian(pixel_value_wrapper_value, psf_params[:]);

  @test_approx_eq ad_grad pixel_value.d[:]
  @test_approx_eq ad_hess[:] pixel_value.h[:]

  # Test the whole least squares function.
  println("Testing psf least squares")

  # Fewer pixels for quick testing.  Also, ForwardDiff.hessian runs into strange
  # problems on the whole image.
  keep_pixels = 20:30

  function evaluate_psf_fit_wrapper_sf{NumType <: Number}(
        psf_param_vec::Vector{NumType}, calculate_derivs::Bool)
    local psf_params = reshape(psf_param_vec, length(PsfParams), 2)
    local squared_error =
      evaluate_psf_fit(psf_params, raw_psf[keep_pixels, keep_pixels], calculate_derivs)
    squared_error
  end

  function evaluate_psf_fit_wrapper_value{NumType <: Number}(psf_param_vec::Vector{NumType})
    local squared_error = evaluate_psf_fit_wrapper_sf(psf_param_vec, false);
    squared_error.v[1]
  end

  squared_error = deepcopy(evaluate_psf_fit_wrapper_sf(psf_params[:], true));

  ad_grad = ForwardDiff.gradient(evaluate_psf_fit_wrapper_value, psf_params[:]);
  ad_hess = ForwardDiff.hessian(evaluate_psf_fit_wrapper_value, psf_params[:]);

  @test_approx_eq ad_grad squared_error.d[:]
  @test_approx_eq ad_hess[:] squared_error.h[:]
end


test_psf_fit()
