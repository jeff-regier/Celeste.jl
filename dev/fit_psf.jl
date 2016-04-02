using Celeste
using Celeste.Types
using Celeste.SkyImages
using Celeste.PSF
using DataFrames

import Celeste.SDSSIO

using Celeste.SensitiveFloats

using ForwardDiff

using Base.Test

using Celeste.Transform

datadir = joinpath(Pkg.dir("Celeste"), "test/data")

using Celeste.PSF

Logging.configure(level=Logging.DEBUG)


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


psf_params_original = PSF.initialize_psf_params(K, for_test=false);
psf_params = deepcopy(psf_params_original)
psf_scale = Float64[1e-2, 1e-2, 1e-1, 1e-1, 1e-1, 1.0]
psf_transform = PSF.get_psf_transform(psf_params; scale=psf_scale);
psf_optimizer = PsfOptimizer(psf_transform, K, verbose=true, grtol=1e-12, ftol=1e-6);
optim_result = psf_optimizer.fit_psf(raw_psf, psf_params_original);

optim_result.iterations
optim_result.f_minimum
optim_result.f_converged
optim_result.gr_converged

psf_params_fit =
  constrain_psf_params(
    unwrap_psf_params(optim_result.minimum), psf_optimizer.psf_transform);


d_dict = Dict()
d_dict["iter"] = Array(Int, 0)
d_dict["interior"] = Array(Bool, 0)
d_dict["delta"] = Array(Float64, 0)
d_dict["f"] = Array(Float64, 0)
for i = 1:length(optim_result.minimum)
  d_dict["x$i"] = Array(Float64, 0)
end
for iter in 1:length(optim_result.trace.states)
  tr = optim_result.trace.states[iter]
  push!(d_dict["f"], tr.value)
  push!(d_dict["iter"], iter)
  push!(d_dict["interior"], tr.metadata["interior"])
  push!(d_dict["delta"], tr.metadata["delta"])
  x = tr.metadata["x"]
  for i = 1:length(x)
    push!(d_dict["x$i"], x[i])
  end
end

DataFrame(d_dict)

se = PSF.evaluate_psf_fit(psf_params_fit, raw_psf, true);
eigvals(se.h)

@time celeste_psf, psf_params_fit =
    PSF.fit_raw_psf_for_celeste(raw_psf, psf_optimizer, psf_params_original)
# @time nm_result = psf_optimizer.fit_psf(raw_psf, psf_params)

# Try with a better init
raw_psf_2 = raw_psf_comp(10., 10.);
@time celeste_psf, psf_params_fit_2 =
    PSF.fit_raw_psf_for_celeste(raw_psf_2, psf_optimizer, psf_params_original)
@time celeste_psf, psf_params_fit_2 =
    PSF.fit_raw_psf_for_celeste(raw_psf_2, psf_optimizer, psf_params_fit)

# Try with a better init
raw_psf_3 = raw_psf_comp(1000., 10.);
@profile celeste_psf, psf_params_fit_3 =
    PSF.fit_raw_psf_for_celeste(raw_psf_3, psf_optimizer, psf_params_fit)
Profile.print(format=:flat)
