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

using Celeste.PSF

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
psf_transform = PSF.get_psf_transform(psf_params);
psf_optimizer = PsfOptimizer(psf_transform, K);

@time nm_result = psf_optimizer.fit_psf(raw_psf, psf_params)
@time nm_result = psf_optimizer.fit_psf(raw_psf, psf_params)
