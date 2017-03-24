using Celeste
using Celeste.Model
using Celeste.SensitiveFloats

import Celeste.PSF: get_psf_at_point, fit_psf,
       get_psf_transform, initialize_psf_params, transform_psf_params!,
       unwrap_psf_params, wrap_psf_params,
       unconstrain_psf_params, constrain_psf_params,
       transform_psf_sensitive_float!,
       PsfOptimizer, fit_raw_psf_for_celeste, trim_psf,
       BivariateNormalDerivatives
import Celeste.Model: eval_psf

using ForwardDiff
using StaticArrays

"""
Evaluate the sum of squared difference between the raw_psf and the psf
represented by psf_params, as well as the derivatives and hessians with
respect to unconstrained parameters.

Returns:
  - A sensitive float for the sum of squared differences.
"""
function evaluate_psf_fit{NumType <: Number}(
    psf_params::Vector{Vector{NumType}}, raw_psf::Matrix{Float64},
    calculate_gradient::Bool)

  K = length(psf_params)
  x_mat = PSF.get_x_matrix_from_psf(raw_psf)

  # TODO: allocate these outside?
  bvn_derivs = BivariateNormalDerivatives{NumType}()
  log_pdf = SensitiveFloat{NumType}(length(PsfParams), 1, true, true)
  pdf = SensitiveFloat{NumType}(length(PsfParams), 1, true, true)

  pixel_value = SensitiveFloat{NumType}(length(PsfParams), K, true, true)
  squared_error = SensitiveFloat{NumType}(length(PsfParams), K, true, true)

  PSF.evaluate_psf_fit!(
      psf_params, raw_psf, x_mat, bvn_derivs,
      log_pdf, pdf, pixel_value, squared_error, calculate_gradient)

  squared_error
end


function load_raw_psf(; x::Float64=500., y::Float64=500.)
  run_num = 3900
  camcol_num = 6
  field_num = 269
  b = 3

  psf_filename =
    @sprintf("%s/%s/%s/%s/psField-%06d-%d-%04d.fit",
        datadir, run_num, camcol_num, field_num,
                 run_num, camcol_num, field_num)
  psf_fits = FITSIO.FITS(psf_filename)
  raw_psf_comp = SDSSIO.read_psf(psf_fits, band_letters[b])
  close(psf_fits)

  eval_psf(raw_psf_comp, x, y)
end


function test_transform_psf_params()
  K = 2
  psf_params = initialize_psf_params(K, for_test=true)
  psf_params_original = deepcopy(psf_params)
  psf_params_free = deepcopy(psf_params)
  psf_transform = PSF.get_psf_transform(psf_params)

  transform_psf_params!(psf_params, psf_params_free, psf_transform, true)
  transform_psf_params!(psf_params, psf_params_free, psf_transform, false)

  psf_params_free_2 = unconstrain_psf_params(psf_params, psf_transform)
  psf_params_2 = constrain_psf_params(psf_params_free, psf_transform)

  for k=1:K
    @test psf_params[k]      ≈ psf_params_original[k]
    @test psf_params_free[k] ≈ psf_params_free_2[k]
    @test psf_params[k]      ≈ psf_params_2[k]
  end
end


function test_psf_fit()
  raw_psf = load_raw_psf()

  # Initialize params
  K = 2
  psf_params = initialize_psf_params(K, for_test=true)
  psf_param_vec = wrap_psf_params(psf_params)

  function pixel_value_wrapper_sf{NumType <: Number}(
      psf_param_vec::Vector{NumType}, calculate_gradient::Bool)

    local psf_params = unwrap_psf_params(psf_param_vec)
    bvn_derivs = BivariateNormalDerivatives{NumType}()
    log_pdf = SensitiveFloat{NumType}(length(PsfParams), 1, true, true)
    pdf = SensitiveFloat{NumType}(length(PsfParams), 1, true, true)

    local pixel_value = SensitiveFloat{NumType}(length(PsfParams), K, true, true)

    local sigma_vec, sig_sf_vec, bvn_vec
    sigma_vec, sig_sf_vec, bvn_vec = PSF.get_sigma_from_params(psf_params)

    # sigma_vec = Vector{Matrix{NumType}}(K)
    # sig_sf_vec = Vector{GalaxySigmaDerivs{NumType}}(K)
    #
    # for k = 1:K
    #   sigma_vec[k] = PSF.get_bvn_cov(psf_params[k][psf_ids.e_axis],
    #                                   psf_params[k][psf_ids.e_angle],
    #                                   psf_params[k][psf_ids.e_scale])
    #   sig_sf_vec[k] = GalaxySigmaDerivs(
    #     psf_params[k][psf_ids.e_angle],
    #     psf_params[k][psf_ids.e_axis],
    #     psf_params[k][psf_ids.e_scale], sigma_vec[k], calculate_gradient)
    #
    # end

    clear!(pixel_value)
    PSF.evaluate_psf_pixel_fit!(
        x, psf_params, sig_sf_vec, bvn_vec,
        bvn_derivs, log_pdf, pdf, pixel_value, calculate_gradient)

    pixel_value
  end

  function pixel_value_wrapper_value{NumType <: Number}(psf_param_vec::Vector{NumType})
    pixel_value_wrapper_sf(psf_param_vec, false).v[]
  end

  x = @SVector [1.0, 2.0]

  sigma_vec = Vector{Matrix{Float64}}(K)
  for k = 1:K
    sigma_vec[k] = PSF.get_bvn_cov(psf_params[k][psf_ids.e_axis],
                                    psf_params[k][psf_ids.e_angle],
                                    psf_params[k][psf_ids.e_scale])
  end

  println("Testing single pixel value")
  psf_components = PsfComponent[
    PsfComponent(psf_params[k][psf_ids.weight], SVector{2,Float64}(psf_params[k][psf_ids.mu]), SMatrix{2,2,Float64,4}(sigma_vec[k]))
                  for k = 1:K ]

  psf_rendered = get_psf_at_point(psf_components, rows=[ x[1] ], cols=[ x[2] ])[1]
  @test psf_rendered ≈ pixel_value_wrapper_value(psf_param_vec)

  pixel_value = deepcopy(pixel_value_wrapper_sf(psf_param_vec, true))

  ad_grad = ForwardDiff.gradient(pixel_value_wrapper_value, psf_param_vec)
  ad_hess = ForwardDiff.hessian(pixel_value_wrapper_value, psf_param_vec)

  @test ad_grad    ≈ pixel_value.d[:]
  @test ad_hess[:] ≈ pixel_value.h[:]

  # Test the whole least squares function.
  println("Testing psf least squares")

  # Fewer pixels for quick testing.  Also, ForwardDiff.hessian runs into strange
  # problems on the whole image.
  keep_pixels = 20:30

  function evaluate_psf_fit_wrapper_sf{NumType <: Number}(
        psf_param_vec::Vector{NumType}, calculate_gradient::Bool)
    local psf_params = unwrap_psf_params(psf_param_vec)
    local squared_error =
      evaluate_psf_fit(psf_params, raw_psf[keep_pixels, keep_pixels], calculate_gradient)
    squared_error
  end

  function evaluate_psf_fit_wrapper_value{NumType <: Number}(psf_param_vec::Vector{NumType})
    local squared_error = evaluate_psf_fit_wrapper_sf(psf_param_vec, false)
    squared_error.v[]
  end

  squared_error = deepcopy(evaluate_psf_fit_wrapper_sf(psf_param_vec, true))

  ad_grad = ForwardDiff.gradient(evaluate_psf_fit_wrapper_value, psf_param_vec)
  ad_hess = ForwardDiff.hessian(evaluate_psf_fit_wrapper_value, psf_param_vec)

  @test ad_grad    ≈ squared_error.d[:]
  @test ad_hess[:] ≈ squared_error.h[:]
end


function test_transform_psf_sensitive_float()
  raw_psf = load_raw_psf()

  K = 2
  psf_params = initialize_psf_params(K, for_test=true)
  psf_transform = PSF.get_psf_transform(psf_params)
  psf_params_free = unconstrain_psf_params(psf_params, psf_transform)
  psf_params_free_vec = wrap_psf_params(psf_params_free)[:]


  # Fewer pixels for quick testing.  Also, ForwardDiff.hessian runs into strange
  # problems on the whole image.
  keep_pixels = 20:30

  function psf_fit_for_optim{NumType <: Number}(
      psf_params_free_vec::Vector{NumType}, calculate_gradient::Bool)

    local sf_free = SensitiveFloat{NumType}(length(PsfParams), K, true, true)
    local psf_params_free = unwrap_psf_params(psf_params_free_vec)
    local psf_params = constrain_psf_params(psf_params_free, psf_transform)
    local sf = evaluate_psf_fit(
      psf_params, raw_psf[keep_pixels, keep_pixels], calculate_gradient)
    transform_psf_sensitive_float!(
      psf_params, psf_transform, sf, sf_free, calculate_gradient)

    sf_free
  end

  function psf_fit_for_optim_val{NumType <: Number}(
      psf_params_free_vec::Vector{NumType})

    psf_fit_for_optim(psf_params_free_vec, false).v[]
  end

  sf_free = deepcopy(psf_fit_for_optim(psf_params_free_vec, true))

  expected_value =
    evaluate_psf_fit(psf_params, raw_psf[keep_pixels, keep_pixels], false).v[]
  ad_grad = ForwardDiff.gradient(psf_fit_for_optim_val, psf_params_free_vec)
  ad_hess = ForwardDiff.hessian(psf_fit_for_optim_val, psf_params_free_vec)

  @test expected_value ≈ sf_free.v[]
  @test sf_free.d[:]   ≈ ad_grad
  @test sf_free.h      ≈ ad_hess
end


function test_psf_optimizer()
  raw_psf = load_raw_psf()

  K = 2
  psf_params = initialize_psf_params(K, for_test=false)
  psf_transform = get_psf_transform(psf_params)
  psf_optimizer = PsfOptimizer(psf_transform, K)

  nm_result = fit_psf(psf_optimizer, raw_psf, psf_params)
  psf_params_fit =
    constrain_psf_params(unwrap_psf_params(Optim.minimizer(nm_result)), psf_transform)

  # Could this test be tighter?
  @test 0.0 < Optim.minimum(nm_result) < 1e-3

  celeste_psf = fit_raw_psf_for_celeste(raw_psf, K)[1]
  rendered_psf = get_psf_at_point(celeste_psf)

  @test Optim.minimum(nm_result) ≈ sum((raw_psf - rendered_psf) .^ 2)

  # Make sure that re-using the optimizer gets the same results.
  raw_psf_10_10 = load_raw_psf(x=10., y=10.)
  celeste_psf_10_10_v1, psf_params_10_10_v1 =
    fit_raw_psf_for_celeste(raw_psf_10_10, K)
  celeste_psf_10_10_v2, psf_params_10_10_v2 =
    fit_raw_psf_for_celeste(raw_psf_10_10, psf_optimizer, psf_params)
  for k=1:K
    @test psf_params_10_10_v1[k] ≈ psf_params_10_10_v2[k]
    for field in fieldnames(celeste_psf_10_10_v1[k])
      @test getfield(celeste_psf_10_10_v1[k], field) ≈ getfield(celeste_psf_10_10_v2[k], field)
    end
  end
end


function test_trim_psf()
    raw_psf = load_raw_psf()
    trim_percent = 0.95
    trimmed_psf = trim_psf(raw_psf; trim_percent=trim_percent)
    @test sum(abs, trimmed_psf) >= trim_percent * sum(abs, raw_psf)
    @test size(trimmed_psf, 1) < size(raw_psf, 1)
    @test size(trimmed_psf, 2) < size(raw_psf, 2)
end


test_transform_psf_sensitive_float()
test_transform_psf_params()
test_psf_fit()
test_psf_optimizer()
test_trim_psf()
