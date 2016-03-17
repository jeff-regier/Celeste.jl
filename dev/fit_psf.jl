using Celeste
using Celeste.Types
using Celeste.SkyImages
using Celeste.PSF

import Celeste.Util
import Celeste.SensitiveFloats
import Celeste.SDSSIO

using PyPlot
using Celeste.SensitiveFloats.SensitiveFloat

using ForwardDiff

using Base.Test


field_dir = joinpath(Pkg.dir("Celeste"), "test/data")

run_num = 4263
camcol_num = 5
field_num = 117

run_str = "004263"
camcol_str = "5"
field_str = "0117"
b = 3
K = 2

psf_filename =
  @sprintf("%s/psField-%06d-%d-%04d.fit", field_dir, run_num, camcol_num, field_num)
psf_fits = FITSIO.FITS(psf_filename);
raw_psf_comp = SDSSIO.read_psf(psf_fits, band_letters[b]);
close(psf_fits)

raw_psf = raw_psf_comp(500., 500.);
#psf = SkyImages.fit_raw_psf_for_celeste(raw_psf);
x_mat = PSF.get_x_matrix_from_psf(raw_psf);

psf_params = initialize_psf_params(K);


using Celeste.Transform.DataTransform
using Celeste.Transform

function transform_psf_sensitive_float!{NumType <: Number}(
    psf_params::Vector{Vector{NumType}}, transform::DataTransform,
    sf::SensitiveFloat{PsfParams, NumType}, sf_free::SensitiveFloat{PsfParams, NumType},
    calculate_derivs::Bool)

  sf_free.v[1] = sf.v[1]
  if calculate_derivs
    K = length(psf_params)
    for k = 1:K
      # This is the diagonal of the Jacobian transform.
      jacobian_diag = zeros(length(PsfParams));

      # These are the hessians of each individual parameter's transform.  We
      # can represent it this way since each parameter's transform only depends on
      # its own value and not on others.
      hessian_values = zeros(length(PsfParams));

      for ind = 1:2
        mu_ind = psf_ids.mu[ind]
        jac, hess =
          Transform.box_derivatives(psf_params[k][mu_ind], psf_transform.bounds[k][:mu][ind]);
        jacobian_diag[mu_ind] = jac
        hessian_values[mu_ind] = hess
      end

      # The rest are one-dimensional.
      for field in setdiff(fieldnames(PsfParams), [ :mu ])
        ind = psf_ids.(field)
        jac, hess =
          Transform.box_derivatives(psf_params[k][1], psf_transform.bounds[k][field][1]);
        jacobian_diag[ind] = jac
        hessian_values[ind] = hess
      end

      hess_inds = (1:length(PsfParams)) + length(PsfParams) * (k - 1)
      sf_free.d[:, k] = jacobian_diag .* sf.d[:, k]
      sf_free.h[hess_inds, hess_inds] =
        ((jacobian_diag * jacobian_diag') .* sf.h[hess_inds, hess_inds]) +
        diagm(hessian_values .* sf_free.d[:, k])
    end
  end

  true # return type
end


using SensitiveFloats


psf_params = initialize_psf_params(K);
psf_params_original = deepcopy(psf_params);
psf_params_free = deepcopy(psf_params);
psf_transform = PSF.get_psf_transform(psf_params);
psf_params_free_vec = wrap_psf_params(psf_params_free)[:];


function psf_fit_for_optim{NumType <: Number}(
    psf_params_free_vec::Vector{NumType}, calculate_derivs::Bool)

  local sf_free = zero_sensitive_float(PsfParams, NumType, K);
  local psf_params_free = unwrap_psf_params(psf_params_free_vec)
  psf_params = unconstrain_psf_params(psf_params_free, psf_transform)
  transform_psf_params!(psf_params, psf_params_free, psf_transform, false)
  local sf = evaluate_psf_fit(psf_params, raw_psf, calculate_derivs);
  transform_psf_sensitive_float!(psf_params, psf_transform, sf, sf_free, calculate_derivs)

  sf_free
end


function psf_fit_for_optim_val{NumType <: Number}(
    psf_params_free_vec::Vector{NumType})

  psf_fit_for_optim(psf_params_free_vec, false).v[1]
end

sf_free = deepcopy(psf_fit_for_optim(psf_params_free_vec, true))

ad_grad = ForwardDiff.gradient(psf_fit_for_optim_val, psf_params_free_vec);
hcat(sf_free.d[:], ad_grad)

psf_fit_for_optim_val
