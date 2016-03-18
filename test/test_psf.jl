using Celeste
using Celeste.Types
using Celeste.SkyImages
using Celeste.BivariateNormals
using Celeste.PSF
using Celeste.SensitiveFloats

import Celeste.Util
import Celeste.SDSSIO

using ForwardDiff

using Base.Test


function test_transform_psf_params()
  K = 2
  psf_params = initialize_psf_params(K, for_test=true);
  psf_params_original = deepcopy(psf_params);
  psf_params_free = deepcopy(psf_params);
  psf_transform = PSF.get_psf_transform(psf_params);

  transform_psf_params!(psf_params, psf_params_free, psf_transform, true);
  transform_psf_params!(psf_params, psf_params_free, psf_transform, false);

  psf_params_free_2 = unconstrain_psf_params(psf_params, psf_transform)
  psf_params_2 = constrain_psf_params(psf_params_free, psf_transform)

  for k=1:K
    @test_approx_eq psf_params[k] psf_params_original[k]
    @test_approx_eq psf_params_free[k] psf_params_free_2[k]
    @test_approx_eq psf_params[k] psf_params_2[k]
  end
end


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
  psf_params = initialize_psf_params(K, for_test=true)
  psf_param_vec = wrap_psf_params(psf_params)

  function pixel_value_wrapper_sf{NumType <: Number}(
      psf_param_vec::Vector{NumType}, calculate_derivs::Bool)

    local psf_params = unwrap_psf_params(psf_param_vec)
    bvn_derivs = BivariateNormalDerivatives{NumType}(NumType);
    log_pdf = zero_sensitive_float(PsfParams, NumType, 1);
    pdf = zero_sensitive_float(PsfParams, NumType, 1);

    local pixel_value = zero_sensitive_float(PsfParams, NumType, K);

    sigma_vec = Array(Matrix{NumType}, K);
    sig_sf_vec = Array(GalaxySigmaDerivs{NumType}, K);

    for k = 1:K
      sigma_vec[k] = Util.get_bvn_cov(psf_params[k][psf_ids.e_axis],
                                      psf_params[k][psf_ids.e_angle],
                                      psf_params[k][psf_ids.e_scale])
      sig_sf_vec[k] = GalaxySigmaDerivs(
        psf_params[k][psf_ids.e_angle],
        psf_params[k][psf_ids.e_axis],
        psf_params[k][psf_ids.e_scale], sigma_vec[k], calculate_tensor=calculate_derivs);

    end

    clear!(pixel_value)
    PSF.evaluate_psf_pixel_fit!(
        x, psf_params, sigma_vec, sig_sf_vec,
        bvn_derivs, log_pdf, pdf, pixel_value, calculate_derivs)

    pixel_value
  end

  function pixel_value_wrapper_value{NumType <: Number}(psf_param_vec::Vector{NumType})
    pixel_value_wrapper_sf(psf_param_vec, false).v[1]
  end

  x = Float64[1.0, 2.0]

  sigma_vec = Array(Matrix{Float64}, K);
  for k = 1:K
    sigma_vec[k] = Util.get_bvn_cov(psf_params[k][psf_ids.e_axis],
                                    psf_params[k][psf_ids.e_angle],
                                    psf_params[k][psf_ids.e_scale])
  end

  println("Testing single pixel value")
  psf_components = PsfComponent[
    PsfComponent(psf_params[k][psf_ids.weight], psf_params[k][psf_ids.mu], sigma_vec[k])
                  for k = 1:K ];

  psf_rendered = get_psf_at_point(psf_components, rows=[ x[1] ], cols=[ x[2] ])[1];
  @test_approx_eq psf_rendered pixel_value_wrapper_value(psf_param_vec)

  pixel_value = deepcopy(pixel_value_wrapper_sf(psf_param_vec, true));

  ad_grad = ForwardDiff.gradient(pixel_value_wrapper_value, psf_param_vec);
  ad_hess = ForwardDiff.hessian(pixel_value_wrapper_value, psf_param_vec);

  @test_approx_eq ad_grad pixel_value.d[:]
  @test_approx_eq ad_hess[:] pixel_value.h[:]

  # Test the whole least squares function.
  println("Testing psf least squares")

  # Fewer pixels for quick testing.  Also, ForwardDiff.hessian runs into strange
  # problems on the whole image.
  keep_pixels = 20:30

  function evaluate_psf_fit_wrapper_sf{NumType <: Number}(
        psf_param_vec::Vector{NumType}, calculate_derivs::Bool)
    local psf_params = unwrap_psf_params(psf_param_vec)
    local squared_error =
      evaluate_psf_fit(psf_params, raw_psf[keep_pixels, keep_pixels], calculate_derivs)
    squared_error
  end

  function evaluate_psf_fit_wrapper_value{NumType <: Number}(psf_param_vec::Vector{NumType})
    local squared_error = evaluate_psf_fit_wrapper_sf(psf_param_vec, false);
    squared_error.v[1]
  end

  squared_error = deepcopy(evaluate_psf_fit_wrapper_sf(psf_param_vec, true));

  ad_grad = ForwardDiff.gradient(evaluate_psf_fit_wrapper_value, psf_param_vec);
  ad_hess = ForwardDiff.hessian(evaluate_psf_fit_wrapper_value, psf_param_vec);

  @test_approx_eq ad_grad squared_error.d[:]
  @test_approx_eq ad_hess[:] squared_error.h[:]
end


function test_least_squares_psf()
  # open FITS file containing PSF for each band
  run = 3900
  camcol = 6
  field = 269

  psf_filename =
    @sprintf("%s/psField-%06d-%d-%04d.fit", datadir, run, camcol, field)
  psf_fits = FITSIO.FITS(psf_filename);
  raw_psf_comp = SDSSIO.read_psf(psf_fits, band_letters[1]);
  close(psf_fits)

  # psf = PSF.get_psf_at_point(500.0, 500.0, raw_psf_comp);
  psf = raw_psf_comp(500., 500.);

  opt_result, mu_vec, sigma_vec, weight_vec =
    PSF.fit_psf_gaussians_least_squares(psf, K=2, ftol=1e-5);

  x_mat = PSF.get_x_matrix_from_psf(psf);
  psf_fit = PSF.render_psf(opt_result.minimum, x_mat);

  @test_approx_eq sum((psf_fit - psf) .^ 2) opt_result.f_minimum
  @test 0 < opt_result.f_minimum < 1e-3

end


function test_psf_transforms()
  mu_vec = Vector{Float64}[ Float64[1, 2], Float64[-1, -2], Float64[1, -1] ]
  sigma_vec = Array(Matrix{Float64}, 3)
  sigma_vec[1] = Float64[ 1 0.1; 0.1 1]
  sigma_vec[2] = Float64[ 1 0.3; 0.3 2]
  sigma_vec[3] = Float64[ 0.5 0.2; 0.2 0.5]
  weight_vec = Float64[0.4, 0.6, 0.1]

  par = PSF.wrap_parameters(mu_vec, sigma_vec, weight_vec)
  mu_vec_test, sigma_vec_test, weight_vec_test = PSF.unwrap_parameters(par)

  for k=1:3
    @test_approx_eq mu_vec_test[k] mu_vec[k]
    @test_approx_eq sigma_vec_test[k] sigma_vec[k]
    @test_approx_eq weight_vec_test[k] weight_vec[k]
  end
end


function test_transform_psf_sensitive_float()
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

  K = 2
  psf_params = initialize_psf_params(K, for_test=true);
  psf_transform = PSF.get_psf_transform(psf_params);
  psf_params_free = unconstrain_psf_params(psf_params, psf_transform);
  psf_params_free_vec = wrap_psf_params(psf_params_free)[:];


  # Fewer pixels for quick testing.  Also, ForwardDiff.hessian runs into strange
  # problems on the whole image.
  keep_pixels = 20:30

  function psf_fit_for_optim{NumType <: Number}(
      psf_params_free_vec::Vector{NumType}, calculate_derivs::Bool)

    local sf_free = zero_sensitive_float(PsfParams, NumType, K);
    local psf_params_free = unwrap_psf_params(psf_params_free_vec)
    local psf_params = constrain_psf_params(psf_params_free, psf_transform)
    local sf = evaluate_psf_fit(
      psf_params, raw_psf[keep_pixels, keep_pixels], calculate_derivs);
    transform_psf_sensitive_float!(
      psf_params, psf_transform, sf, sf_free, calculate_derivs)

    sf_free
  end

  function psf_fit_for_optim_val{NumType <: Number}(
      psf_params_free_vec::Vector{NumType})

    psf_fit_for_optim(psf_params_free_vec, false).v[1]
  end

  sf_free = deepcopy(psf_fit_for_optim(psf_params_free_vec, true));

  expected_value =
    evaluate_psf_fit(psf_params, raw_psf[keep_pixels, keep_pixels], false).v[1];
  ad_grad = ForwardDiff.gradient(psf_fit_for_optim_val, psf_params_free_vec);
  ad_hess = ForwardDiff.hessian(psf_fit_for_optim_val, psf_params_free_vec);

  @test_approx_eq expected_value sf_free.v[1]
  @test_approx_eq sf_free.d[:] ad_grad
  @test_approx_eq sf_free.h ad_hess
end


test_transform_psf_sensitive_float()
test_transform_psf_params()
test_psf_transforms()
test_psf_fit()
test_least_squares_psf()
