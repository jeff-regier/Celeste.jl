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

# Initialize params
psf_params = Array(Vector{Float64}, K)
for k=1:K
  psf_params[k] = zeros(length(PsfParams))
  psf_params[k][psf_ids.mu] = [0., 0.]
  psf_params[k][psf_ids.e_axis] = 0.8
  psf_params[k][psf_ids.e_angle] = pi / 4
  psf_params[k][psf_ids.e_scale] = sqrt(2 * k)
  psf_params[k][psf_ids.weight] = 1/ K
end

sf = evaluate_psf_fit(psf_params, raw_psf, true);
sf_free = deepcopy(sf);

k = 1
using Celeste.Transform
psf_transform = PSF.get_psf_transform(psf_params);

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
