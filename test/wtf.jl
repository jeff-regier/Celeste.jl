using Celeste
using CelesteTypes
using Base.Test
using Distributions
using SampleData
using Transform
using PyPlot

using ForwardDiff

import Synthetic
import WCS

println("Running hessian tests.")

# TODO: test with a real and asymmetric wcs jacobian.
blob, mp, three_bodies = gen_three_body_dataset();
omitted_ids = Int64[];
kept_ids = setdiff(1:length(ids), omitted_ids);

s = 1
b = 3

patch = mp.patches[s, b];
u = mp.vp[s][ids.u]
u_pix = WCS.world_to_pixel(
  patch.wcs_jacobian, patch.center, patch.pixel_center, u)
x = ceil(u_pix + [1.0, 2.0])

elbo_vars = ElboDeriv.ElboIntermediateVariables(Float64, 1);

psf_ind = 1
gal_ind = 4
gal_type = 2
gcc_ind = (psf_ind, gal_ind, gal_type, s)

star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(mp, b);

# Raw:
gcc = gal_mcs[gcc_ind...];
py1, py2, f_pre = ElboDeriv.eval_bvn_pdf(gcc.bmc, x)


# And now through fs1m
# The pixel and world centers shouldn't matter for derivatives.
psf = patch.psf[psf_ind];

# Pick out a single galaxy component for testing.
gp = galaxy_prototypes[gal_type][gal_ind];
e_dev_dir = gal_type == 1 ? 1.0 : -1.0
e_dev = mp.vp[s][ids.e_dev];
e_dev_i = gal_type == 1 ? e_dev : 1 - e_dev

gcc2 = ElboDeriv.GalaxyCacheComponent(
         e_dev_dir, e_dev_i, gp, psf, u_pix,
         mp.vp[s][ids.e_axis], mp.vp[s][ids.e_angle], mp.vp[s][ids.e_scale]);

py1, py2, f_pre2 = ElboDeriv.eval_bvn_pdf(gcc2.bmc, x);

f_pre
f_pre2















# ok
