using Celeste
using Celeste.Types
using Celeste.SkyImages
using Celeste.PSF

import Celeste.Util
import Celeste.SensitiveFloats
import Celeste.SDSSIO

using Celeste.SensitiveFloats.SensitiveFloat

using ForwardDiff

using Base.Test


datadir = joinpath(Pkg.dir("Celeste"), "test/data")

run_num = 4263
camcol_num = 5
field_num = 117

run_str = "004263"
camcol_str = "5"
field_str = "0117"
b = 3
K = 2

psf_filename =
  @sprintf("%s/psField-%06d-%d-%04d.fit", datadir, run_num, camcol_num, field_num)
psf_fits = FITSIO.FITS(psf_filename);
raw_psf_comp = SDSSIO.read_psf(psf_fits, band_letters[b]);
close(psf_fits)

raw_psf = raw_psf_comp(500., 500.);

psf_params = initialize_psf_params(K);
psf_params_original = deepcopy(psf_params);
psf_params_free = deepcopy(psf_params);
psf_transform = PSF.get_psf_transform(psf_params);
psf_params_free_vec = wrap_psf_params(psf_params_free)[:];


function psf_fit_for_optim{NumType <: Number}(
    psf_params_free_vec::Vector{NumType}, calculate_derivs::Bool)

  local sf_free = zero_sensitive_float(PsfParams, NumType, K);
  local psf_params_free = unwrap_psf_params(psf_params_free_vec)
  local psf_params = unconstrain_psf_params(psf_params_free, psf_transform)
  local sf = evaluate_psf_fit(psf_params, raw_psf, calculate_derivs);
  transform_psf_sensitive_float!(psf_params, psf_transform, sf, sf_free, calculate_derivs)

  sf_free
end


function psf_fit_for_optim_val{NumType <: Number}(
    psf_params_free_vec::Vector{NumType})

  psf_fit_for_optim(psf_params_free_vec, false).v[1]
end

sf_free = deepcopy(psf_fit_for_optim(psf_params_free_vec, true));

ad_grad = ForwardDiff.gradient(psf_fit_for_optim_val, psf_params_free_vec);
ad_hess = ForwardDiff.hessian(psf_fit_for_optim_val, psf_params_free_vec);

@test_approx_eq sf_free.d[:] ad_grad
@test_approx_eq sf_free.h ad_hess
