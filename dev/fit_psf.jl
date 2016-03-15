using Celeste
using Celeste.Types
using Celeste.SkyImages
using Celeste.BivariateNormals
import Celeste.Util
import Celeste.SensitiveFloats

using SloanDigitalSkySurvey.PSF
using PyPlot


field_dir = joinpath(Pkg.dir("Celeste"), "test/data")

run_num = "004263"
camcol_num = "5"
field_num = "0117"
b = 3
sf = SensitiveFloats.zero_sensitive_float(Types.PsfParams, Float64);


psf_fname = "$field_dir/psField-$run_num-$camcol_num-$field_num.fit"
@assert(isfile(psf_fname), "Cannot find mask file $(psf_fname)")
raw_psf_comp = SkyImages.load_sdss_psf(psf_fname, b);

raw_psf = get_psf_at_point(500., 500., raw_psf_comp);
psf = SkyImages.fit_raw_psf_for_celeste(raw_psf);
x_mat = PSF.get_x_matrix_from_psf(raw_psf);
wcs_jacobian = eye(2);



mu = Float64[1.0, 2.0]
w = 0.5
e_axis = 0.5
e_angle = pi / 4
e_scale = 10.0

sigma = Util.get_bvn_cov(e_axis, e_angle, e_scale)
sig_sf = GalaxySigmaDerivs(
  e_angle, e_axis, e_scale, sigma, calculate_tensor=true);

# Note: bvn is immutable and so must be redefined at each iteration
bvn = BvnComponent{Float64}(mu, sigma, w);
bvn_derivs = BivariateNormalDerivatives{Float64}(Float64);

bvn_pdf = zeros(size(x_mat));

for i=1:length(x_mat)
  eval_bvn_pdf!(bvn_derivs, bvn, x_mat[i])
  get_bvn_derivs!(bvn_derivs, bvn, true, true)
  transform_bvn_derivs!(bvn_derivs, sig_sf, wcs_jacobian, true)
  bvn_pdf[i] = bvn_derivs.f_pre[1]
end
matshow(bvn_pdf)
