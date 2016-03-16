using Celeste
using Celeste.Types
using Celeste.SkyImages
using Celeste.BivariateNormals
import Celeste.Util
import Celeste.SensitiveFloats
import Celeste.SDSSIO

using SloanDigitalSkySurvey.PSF
using PyPlot
using Celeste.SensitiveFloats.SensitiveFloat

using ForwardDiff

using Base.Test

include("src/PSF.jl")


field_dir = joinpath(Pkg.dir("Celeste"), "test/data")

run_num = 4263
camcol_num = 5
field_num = 117

run_str = "004263"
camcol_str = "5"
field_str = "0117"
b = 3

psf_filename =
  @sprintf("%s/psField-%06d-%d-%04d.fit", field_dir, run_num, camcol_num, field_num)
psf_fits = FITSIO.FITS(psf_filename);
raw_psf_comp = SDSSIO.read_psf(psf_fits, band_letters[b]);
close(psf_fits)


raw_psf = raw_psf_comp(500., 500.);
#psf = SkyImages.fit_raw_psf_for_celeste(raw_psf);
x_mat = PSF.get_x_matrix_from_psf(raw_psf);
wcs_jacobian = eye(2);

K = 2

# Initialize params
psf_params = zeros(length(Types.PsfParams), K)
for k=1:K
  psf_params[psf_ids.mu, k] = [0., 0.]
  psf_params[psf_ids.e_axis, k] = 0.8
  psf_params[psf_ids.e_angle, k] = pi / 4
  psf_params[psf_ids.e_scale, k] = sqrt(2 * k)
  psf_params[psf_ids.weight, k] = 1/ K
end

# For debugging
NumType = Float64



function pixel_value_wrapper_sf{NumType <: Number}(psf_param_vec::Vector{NumType})

  psf_params = reshape(psf_param_vec, length(PsfParams), 2)

  bvn_derivs = BivariateNormalDerivatives{NumType}(NumType);
  log_pdf = SensitiveFloats.zero_sensitive_float(PsfParams, NumType, 1);
  pdf = SensitiveFloats.zero_sensitive_float(PsfParams, NumType, 1);

  pixel_value = SensitiveFloats.zero_sensitive_float(PsfParams, NumType, K);
  squared_error = SensitiveFloats.zero_sensitive_float(PsfParams, NumType, K);

  sigma_vec = Array(Matrix{NumType}, K);
  sig_sf_vec = Array(GalaxySigmaDerivs{NumType}, K);

  for k = 1:K
    sigma_vec[k] = Util.get_bvn_cov(psf_params[psf_ids.e_axis, k],
                                    psf_params[psf_ids.e_angle, k],
                                    psf_params[psf_ids.e_scale, k])
    sig_sf_vec[k] = GalaxySigmaDerivs(
      psf_params[psf_ids.e_angle, k],
      psf_params[psf_ids.e_axis, k],
      psf_params[psf_ids.e_scale, k], sigma_vec[k], calculate_tensor=true);

  end

  clear!(pixel_value)
  evaluate_psf_pixel_fit!(
      x, psf_params, sigma_vec, sig_sf_vec,
      bvn_derivs, log_pdf, pdf, pixel_value)

  pixel_value
end

function pixel_value_wrapper_value{NumType <: Number}(psf_param_vec::Vector{NumType})
  pixel_value_wrapper_sf(psf_param_vec).v[1]
end


# Pick a point for testing
x_ind = 1508
x = x_mat[x_ind]

psf_components = PsfComponent[
  PsfComponent(psf_params[psf_ids.weight, k], psf_params[psf_ids.mu, k], sigma_vec[k])
                for k = 1:K ];

psf_rendered = get_psf_at_point(psf_components, rows=[ x[1] ], cols=[ x[2] ])[1];
@test_approx_eq psf_rendered pixel_value_wrapper_value(psf_params[:])

pixel_value = pixel_value_wrapper_sf(psf_params[:]);

ad_grad = ForwardDiff.gradient(pixel_value_wrapper_value, psf_params[:])
ad_hess = ForwardDiff.hessian(pixel_value_wrapper_value, psf_params[:])

hcat(ad_grad, pixel_value.d[:])
hcat(ad_hess[:], pixel_value.h[:])

#matshow(psf_image)
