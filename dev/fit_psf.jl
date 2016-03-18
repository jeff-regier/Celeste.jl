using Celeste
using Celeste.Types
using Celeste.SkyImages
using Celeste.PSF

import Celeste.Util
import Celeste.SDSSIO

using Celeste.SensitiveFloats

using ForwardDiff

using Base.Test

using Celeste.Transform

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

using Celeste.BivariateNormals.BivariateNormalDerivatives
import Optim

using Celeste.PSF.get_x_matrix_from_psf
using Celeste.PSF.evaluate_psf_fit!

# Only include until this is merged with Optim.jl.
include("src/newton_trust_region.jl")

type PsfOptimizer
  psf_transform::DataTransform
  ftol::Float64
  grtol::Float64
  num_iters::Int
  raw_psf::Matrix{Float64}
  K::Int

  # Variable that will be allocated in optimization:
  x_mat::Matrix{Float64}
  bvn_derivs::BivariateNormalDerivatives{Float64};

  log_pdf::SensitiveFloat{PsfParams, Float64};
  pdf::SensitiveFloat{PsfParams, Float64};
  pixel_value::SensitiveFloat{PsfParams, Float64};
  squared_error::SensitiveFloat{PsfParams, Float64};
  sf_free::SensitiveFloat{PsfParams, Float64};

  psf_params_free_vec_cache::Vector{Float64}

  # functions
  psf_2df::Optim.TwiceDifferentiableFunction
  fit_psf::Function

  function PsfOptimizer(psf_transform::DataTransform, K::Int)

    ftol = grtol = 1e-9
    num_iters = 50

    bvn_derivs = BivariateNormalDerivatives{Float64}(Float64);

    log_pdf = SensitiveFloats.zero_sensitive_float(PsfParams, Float64, 1);
    pdf = SensitiveFloats.zero_sensitive_float(PsfParams, Float64, 1);
    pixel_value = SensitiveFloats.zero_sensitive_float(PsfParams, Float64, K);
    squared_error = SensitiveFloats.zero_sensitive_float(PsfParams, Float64, K);
    sf_free = SensitiveFloats.zero_sensitive_float(PsfParams, Float64, K);

    x_mat = Array(Float64, 0, 0)
    raw_psf = Array(Float64, 0, 0)
    psf_params_free_vec_cache = fill(NaN, K * length(PsfParams))

    function psf_fit_for_optim{NumType <: Number}(
        psf_params_free_vec::Vector{NumType})

      if psf_params_free_vec == psf_params_free_vec_cache
        return sf_free
      else
        psf_params_free_vec_cache = deepcopy(psf_params_free_vec)
      end
      psf_params_free = unwrap_psf_params(psf_params_free_vec)
      psf_params = constrain_psf_params(psf_params_free, psf_transform)

      # Update squared_error in place.
      evaluate_psf_fit!(
          psf_params, raw_psf, x_mat, bvn_derivs,
          log_pdf, pdf, pixel_value, squared_error, true)

      # Update sf_free in place.
      transform_psf_sensitive_float!(
        psf_params, psf_transform, squared_error, sf_free, true)

      sf_free
    end

    function psf_fit_value{NumType <: Number}(psf_params_free_vec::Vector{NumType})
      psf_fit_for_optim(psf_params_free_vec).v[1]
    end

    function psf_fit_grad!(
        psf_params_free_vec::Vector{Float64}, grad::Vector{Float64})
      grad[:] = psf_fit_for_optim(psf_params_free_vec).d[:]
    end

    function psf_fit_hess!(
        psf_params_free_vec::Vector{Float64}, hess::Matrix{Float64})
      hess[:] = psf_fit_for_optim(psf_params_free_vec).h
      hess[:] = 0.5 * (hess + hess')
    end

    psf_2df = Optim.TwiceDifferentiableFunction(
      psf_fit_value, psf_fit_grad!, psf_fit_hess!)

    function fit_psf(psf::Matrix{Float64}, initial_params::Vector{Vector{Float64}})
      raw_psf = psf
      x_mat = get_x_matrix_from_psf(raw_psf);
      psf_params_free = unconstrain_psf_params(psf_params, psf_transform);
      psf_params_free_vec = vec(wrap_psf_params(psf_params_free));
      nm_result = newton_tr(psf_2df,
                            psf_params_free_vec,
                            xtol = 0.0,
                            grtol = grtol,
                            ftol = ftol,
                            iterations = num_iters,
                            store_trace = false,
                            show_trace = false,
                            extended_trace = false,
                            initial_delta=10.0,
                            delta_hat=1e9,
                            rho_lower = 0.2)
      nm_result
    end

    new(psf_transform, ftol, grtol, num_iters, raw_psf, K,
      x_mat, bvn_derivs, log_pdf, pdf, pixel_value, squared_error, sf_free,
      psf_params_free_vec_cache, psf_2df, fit_psf)
  end
end


psf_params_original = PSF.initialize_psf_params(K, for_test=false);
psf_params = deepcopy(psf_params_original)
psf_transform = PSF.get_psf_transform(psf_params);
psf_optimizer = PsfOptimizer(psf_transform, K);

nm_result = psf_optimizer.fit_psf(raw_psf, psf_params)
























include("src/PSF.jl")



psf_params_original = PSF.initialize_psf_params(K, for_test=false);
psf_params = deepcopy(psf_params_original)
psf_transform = PSF.get_psf_transform(psf_params);
psf_params_free = unconstrain_psf_params(psf_params, psf_transform)
psf_params_free_vec = vec(wrap_psf_params(psf_params_free));


max_iters = 50
verbose = true
rho_lower = 0.2

nm_result = newton_tr(d,
                      psf_params_free_vec,
                      xtol = 0.0,
                      grtol = 1e-16,
                      ftol = 1e-16,
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
