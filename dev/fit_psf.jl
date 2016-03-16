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



using Celeste.Transform.ParamBounds
using Celeste.Transform.ParamBox
using Celeste.Transform.DataTransform

function get_psf_transform(psf_params::Vector{Vector{Float64}})

  bounds = Array(ParamBounds, size(psf_params, 2))

  # Note that, for numerical reasons, the bounds must be on the scale
  # of reasonably meaningful changes.
  for k in 1:K
    bounds[k] = ParamBounds()
    bounds[k][:mu] = fill(ParamBox(-5.0, 5.0, 1.0), 2)
    bounds[k][:e_axis] = ParamBox[ ParamBox(0.1, 1.0, 1.0) ]
    bounds[k][:e_angle] = ParamBox[ ParamBox(0.0, 4 * pi, 1.0) ]
    bounds[k][:e_scale] = ParamBox[ ParamBox(0.25, Inf, 1.0) ]

    # Note that the weights do not need to sum to one.
    bounds[k][:e_weight] = ParamBox[ ParamBox(0.05, 2.0, 1.0) ]
  end
  DataTransform(bounds, active_sources=collect(1:K), S=K)
end


psf_transform = get_psf_transform(psf_params)

get_transform_derivatives
