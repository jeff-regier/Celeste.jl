using Celeste
using Celeste.Types
using Celeste.SkyImages
using Celeste.BivariateNormals
import Celeste.Util
import Celeste.SensitiveFloats
import Celeste.SDSSIO

using SloanDigitalSkySurvey.PSF
using PyPlot


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
psf_fit = SensitiveFloats.zero_sensitive_float(PsfParams, Float64, K);


# Initialize params
psf_params = zeros(length(Types.PsfParams), K)
for k=1:K
  psf_params[psf_ids.mu, k] = [0., 0.]
  psf_params[psf_ids.e_axis, k] = 0.8
  psf_params[psf_ids.e_angle, k] = pi / 4
  psf_params[psf_ids.e_scale, k] = sqrt(2 * k)
  psf_params[psf_ids.weight, k] = 1/ K
end

NumType = Float64

# Functions
bvn_derivs = BivariateNormalDerivatives{Float64}(Float64);

sigma_vec = Array(Matrix{Float64}, K);
for k = 1:K
  sigma_vec[k] = Util.get_bvn_cov(psf_params[psf_ids.e_axis, k],
                                  psf_params[psf_ids.e_angle, k],
                                  psf_params[psf_ids.e_scale, k])
end


# Get the value of the log pdf at one point
log_pdf = SensitiveFloats.zero_sensitive_float(PsfParams, Float64, 1);
pdf = SensitiveFloats.zero_sensitive_float(PsfParams, Float64, K);

x = x_mat[29, 28]

for k = 1:K
  bvn = BvnComponent{NumType}(
    psf_params[psf_ids.mu, k], sigma_vec[k], psf_params[psf_ids.weight, k]);
  eval_bvn_pdf!(bvn_derivs, bvn, x)
  get_bvn_derivs!(bvn_derivs, bvn, true, true)
  sig_sf = GalaxySigmaDerivs(
    psf_params[psf_ids.e_angle, k],
    psf_params[psf_ids.e_axis, k],
    psf_params[psf_ids.e_scale, k], sigma_vec[k], calculate_tensor=true);
  transform_bvn_derivs!(bvn_derivs, sig_sf, wcs_jacobian, true)

  log_pdf.v[1] = bvn_derivs.f_pre[1]
  log_pdf.d[psf_ids.mu] = bvn_derivs.bvn_u_d
  log_pdf.d[[psf_ids.e_axis, psf_ids.e_angle, psf_ids.e_scale]] =
    bvn_derivs.bvn_s_d

end




function evaluate_pdf!{NumType <: Number}(
    psf_fit::SensitiveFloat{NumType}, x_mat::Array{Vector{Float64, 1}, 2})

  K =
  for i=1:length(x_mat)
    for k=1:
    eval_bvn_pdf!(bvn_derivs, bvn, x_mat[i])
    get_bvn_derivs!(bvn_derivs, bvn, true, true)
    transform_bvn_derivs!(bvn_derivs, sig_sf, wcs_jacobian, true)
    bvn_pdf[i] = bvn_derivs.f_pre[1]
  end
end

mu = Float64[1.0, 2.0]
w = 0.5
e_axis = 0.5
e_angle = pi / 4
e_scale = 10.0



bvn_pdf = zeros(size(x_mat));

for i=1:length(x_mat)
  eval_bvn_pdf!(bvn_derivs, bvn, x_mat[i])
  get_bvn_derivs!(bvn_derivs, bvn, true, true)
  transform_bvn_derivs!(bvn_derivs, sig_sf, wcs_jacobian, true)
  bvn_pdf[i] = bvn_derivs.f_pre[1]
end
matshow(bvn_pdf)
