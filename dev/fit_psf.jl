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

using Celeste.Transform
using Celeste.Transform.box_parameter
using Celeste.Transform.unbox_parameter

function transform_psf_sensitive_float!{NumType <: Number}(
    psf_params::Vector{Vector{Float64}}, transform::DataTransform,
    sf::SensitiveFloat{NumType}, sf_free::SensitiveFloat{NumType},
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


function transform_psf_params!{NumType <: Number}(
    psf_params::Vector{Vector{NumType}}, psf_params_free::Vector{Vector{NumType}},
    psf_transform::DataTransform, to_unconstrained::Bool)

  for k=1:length(psf_params)
    for (param, constraint_vec) in psf_transform.bounds[k]
      for ind in 1:length(psf_ids.(param))
        constraint = constraint_vec[ind]
        to_unconstrained ?
          psf_params_free[k][ind] = unbox_parameter(psf_params[k][ind], constraint):
          psf_params[k][ind] = box_parameter(psf_params_free[k][ind], constraint)
      end
    end
  end

  true # return type
end




function psf_fit_for_optim{NumType <: Number}(psf_params_vec::Vector{NumType})
  psf_array_to_params!()
end

sf = evaluate_psf_fit(psf_params, raw_psf, true);
sf_free = deepcopy(sf);
psf_transform = PSF.get_psf_transform(psf_params);

psf_params_original = deepcopy(psf_params);
psf_params_free = deepcopy(psf_params);

transform_psf_params!(psf_params, psf_params_free, psf_transform, true);
transform_psf_params!(psf_params, psf_params_free, psf_transform, false);
