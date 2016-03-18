using Celeste
using Celeste.Types
using Celeste.SkyImages
using Celeste.PSF

import Celeste.Util
import Celeste.SDSSIO

using Celeste.SensitiveFloats

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





include("src/PSF.jl")

function psf_fit_for_optim{NumType <: Number}(
    psf_params_free_vec::Vector{NumType}, calculate_derivs::Bool)

  sf_free = zero_sensitive_float(PsfParams, NumType, K);
  psf_params_free = unwrap_psf_params(psf_params_free_vec)
  psf_params = constrain_psf_params(psf_params_free, psf_transform)
  sf = evaluate_psf_fit(psf_params, raw_psf, calculate_derivs);
  if verbose
    # println("------------------- Params:")
    # println(psf_params)
    # println(psf_params_free)
    # println(PSF.get_sigma_from_params(psf_params)[1])
    # println("------------------- ok ok ")
  end
  transform_psf_sensitive_float!(
    psf_params, psf_transform, sf, sf_free, calculate_derivs)

  sf_free
end

function psf_fit_value{NumType <: Number}(psf_params_free_vec::Vector{NumType})
  psf_fit_for_optim(psf_params_free_vec, false).v[1]
end

function psf_fit_grad!(
    psf_params_free_vec::Vector{Float64}, grad::Vector{Float64})
  grad[:] = psf_fit_for_optim(psf_params_free_vec, true).d[:]
end

function psf_fit_hess!(
    psf_params_free_vec::Vector{Float64}, hess::Matrix{Float64})
  hess[:] = psf_fit_for_optim(psf_params_free_vec, true).h
  hess[:] = 0.5 * (hess + hess')
end

d = Optim.TwiceDifferentiableFunction(
  psf_fit_value, psf_fit_grad!, psf_fit_hess!)

# Only include until this is merged with Optim.jl.
include("src/newton_trust_region.jl")

psf_params_original = PSF.initialize_psf_params(K, for_test=false);
psf_params = deepcopy(psf_params_original)
psf_transform = PSF.get_psf_transform(psf_params);
psf_params_free = unconstrain_psf_params(psf_params, psf_transform)
psf_params_free_vec = wrap_psf_params(psf_params_free)[:];


max_iters = 50
verbose = true
rho_lower = 0.2

nm_result = newton_tr(d,
                      psf_params_free_vec,
                      xtol = 0.0,
                      ftol = 1e-9,
                      grtol = 1e-9,
                      iterations = max_iters,
                      store_trace = false,
                      show_trace = false,
                      extended_trace = verbose,
                      initial_delta=10.0,
                      delta_hat=1e9,
                      rho_lower = rho_lower)
nm_result.f_minimum

psf_params_fit =
  constrain_psf_params(unwrap_psf_params(nm_result.minimum), psf_transform)
PSF.get_sigma_from_params(psf_params_fit)[1]
psf_params_free_vec_fit =
  wrap_psf_params(unconstrain_psf_params(psf_params_fit, psf_transform));

sf = evaluate_psf_fit(psf_params, raw_psf, true);
diag(sf.h)
hess = zeros(length(psf_params_free_vec), length(psf_params_free_vec));
psf_fit_hess!(psf_params_free_vec_fit, hess);
diag(hess)
hess_old
