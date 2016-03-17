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

    # These are the hessians of each individual parameter's transform.  We
    # can represent it this way since each parameter's transform only depends on
    # its own value and not on others.

    # This is the diagonal of the Jacobian transform.
    jacobian_diag = zeros(length(PsfParams) * K);

    # These are hte Hessians of the each parameter's transform.
    hessian_values = zeros(length(PsfParams) * K);

    for k = 1:K
      offset = length(PsfParams) * (k - 1)
      for ind = 1:2
        mu_ind = psf_ids.mu[ind]
        jac, hess =
          Transform.box_derivatives(psf_params[k][mu_ind], psf_transform.bounds[k][:mu][ind]);
        jacobian_diag[offset + mu_ind] = jac
        hessian_values[offset + mu_ind] = hess
      end

      # The rest are one-dimensional.
      for field in setdiff(fieldnames(PsfParams), [ :mu ])
        ind = psf_ids.(field)
        jac, hess =
          Transform.box_derivatives(psf_params[k][ind], psf_transform.bounds[k][field][1]);
        jacobian_diag[offset + ind] = jac
        hessian_values[offset + ind] = hess
      end
    end

    # Apply the transformations.
    sf_free.d = reshape(jacobian_diag .* sf.d[:], length(PsfParams), K)

    # Calculate the Hessian
    sf_free.h =
      ((jacobian_diag * jacobian_diag') .* sf.h) +
      diagm(hessian_values .* sf.d[:])
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
  # local sf = zero_sensitive_float(PsfParams, NumType, K);
  local psf_params_free = unwrap_psf_params(psf_params_free_vec)
  local psf_params = unconstrain_psf_params(psf_params_free, psf_transform)

  local sf = evaluate_psf_fit(psf_params, raw_psf, calculate_derivs);
  # sf.v[1] = 0.0
  # for k=1:length(psf_params)
  #   sf.v[1] += 0.5 * vecdot(psf_params[k], psf_params[k])
  #   if calculate_derivs
  #     sf.d[:, k] = psf_params[k]
  #   end
  # end
  # if calculate_derivs
  #   sf.h = eye(length(psf_params_free_vec))
  # end

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
