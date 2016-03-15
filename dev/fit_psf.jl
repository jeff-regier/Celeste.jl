using Celeste
using Celeste.Types
using Celeste.SkyImages
using Celeste.BivariateNormals
import Celeste.Util
import Celeste.SensitiveFloats

using SloanDigitalSkySurvey.PSF
using PyPlot


field_dir = joinpath(Pkg.dir("Celeste"), "test/data")

run_num = "004263"
camcol_num = "5"
field_num = "0117"
b = 3

psf_fname = "$field_dir/psField-$run_num-$camcol_num-$field_num.fit"
@assert(isfile(psf_fname), "Cannot find mask file $(psf_fname)")
raw_psf_comp = SkyImages.load_sdss_psf(psf_fname, b);

raw_psf = get_psf_at_point(500., 500., raw_psf_comp);
#psf = SkyImages.fit_raw_psf_for_celeste(raw_psf);
x_mat = PSF.get_x_matrix_from_psf(raw_psf);
wcs_jacobian = eye(2);

K = 2
psf_fit = SensitiveFloats.zero_sensitive_float(PsfParams, Float64, K);
log_pdf = SensitiveFloats.zero_sensitive_float(PsfParams, Float64, K);


# Initialize params
psf_params = zeros(length(Types.PsfParams), K)
for k=1:K
  psf_params[psf_ids.mu, k] = [0., 0.]
  psf_params[psf_ids.e_axis, k] = 1.0
  psf_params[psf_ids.e_angle, k] = 0.0
  psf_params[psf_ids.e_scale, k] = sqrt(k)
  psf_params[psf_ids.weight, k] = 1/ K
end

# Functions

function evaluate_log_pdf!{NumType <: Number}(
    log_pdf::SensitiveFloat{NumType}, x_mat::Array{Vector{Float64, 1}, 2},
    psf_params::Matrix{NumType})

  bvn_derivs = BivariateNormalDerivatives{Float64}(Float64);
  sig_sf = GalaxySigmaDerivs(
    e_angle, e_axis, e_scale, sigma, calculate_tensor=true);

  K = size(psf_params, 2)
  for k=1:K
    sigma = Util.get_bvn_cov(psf_params[psf_ids.e_axis, k],
                             psf_params[psf_ids.e_angle, k],
                             psf_params[psf_ids.e_scale, k])

    # Note: bvn is immutable and so must be redefined at each iteration
    bvn = BvnComponent{NumType}(mu, sigma, w);

    eval_bvn_pdf!(bvn_derivs, bvn, x_mat[i])
    get_bvn_derivs!(bvn_derivs, bvn, true, true)
    transform_bvn_derivs!(bvn_derivs, sig_sf, wcs_jacobian, true)
    bvn_pdf[i] = bvn_derivs.f_pre[1]
  end
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
