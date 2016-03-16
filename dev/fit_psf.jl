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
  psf_params[k] = zeros(length(Types.PsfParams))
  psf_params[k][psf_ids.mu] = [0., 0.]
  psf_params[k][psf_ids.e_axis] = 0.8
  psf_params[k][psf_ids.e_angle] = pi / 4
  psf_params[k][psf_ids.e_scale] = sqrt(2 * k)
  psf_params[k][psf_ids.weight] = 1/ K
end
