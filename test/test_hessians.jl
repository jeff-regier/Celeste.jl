using Celeste
using CelesteTypes
using Base.Test
using Distributions
using SampleData
using Transform
#using PyPlot

using ForwardDiff

import Synthetic
import WCS

println("Running hessian tests.")

function test_fsXm_derivatives()
  # TODO: test with a real and asymmetric wcs jacobian.
  blob, mp, three_bodies = gen_three_body_dataset();
  omitted_ids = Int64[];
  kept_ids = setdiff(1:length(ids), omitted_ids);

  s = 1
  b = 3

  patch = mp.patches[s];
  u = mp.vp[s][ids.u]
  u_pix = WCS.world_to_pixel(
    patch.wcs_jacobian, patch.center, patch.pixel_center, u)
  x = ceil(u_pix + [1.0, 2.0])

  elbo_vars = ElboDeriv.ElboIntermediateVariables(Float64);

  ###########################
  # Galaxies

  # Pick out a single galaxy component for testing.
  # The index is psf, galaxy, gal type, source
  gcc_ind = (1, 1, 1, s)
  function f_wrap_gal{T <: Number}(par::Vector{T})
    # This uses mp, x, wcs_jacobian, and gcc_ind from the enclosing namespace.
    if T != Float64
      mp_fd = CelesteTypes.forward_diff_model_params(T, mp);
    else
      mp_fd = deepcopy(mp)
    end
    fs1m = zero_sensitive_float(GalaxyPosParams, T, 1);

    # Make sure par is as long as the galaxy parameters.
    @assert length(par) == length(shape_standard_alignment[2])
    for p1 in 1:length(par)
        p0 = shape_standard_alignment[2][p1]
        mp_fd.vp[s][p0] = par[p1]
    end
    star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(mp_fd, b);
    elbo_vars_fd = ElboDeriv.ElboIntermediateVariables(T);
    ElboDeriv.accum_galaxy_pos!(
      elbo_vars_fd, gal_mcs[gcc_ind...], x, fs1m, patch.wcs_jacobian);
    fs1m.v
  end

  function mp_to_par_gal(mp::ModelParams{Float64})
    par = zeros(length(shape_standard_alignment[2]))
    for p1 in 1:length(par)
        p0 = shape_standard_alignment[2][p1]
        par[p1] = mp.vp[s][p0]
    end
    par
  end

  par = mp_to_par_gal(mp);

  fs1m = zero_sensitive_float(GalaxyPosParams, 1);
  star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(mp, b);
  ElboDeriv.accum_galaxy_pos!(
    elbo_vars, gal_mcs[gcc_ind...], x, fs1m, patch.wcs_jacobian);

  # Two sanity checks.
  gcc = gal_mcs[gcc_ind...];
  v = ElboDeriv.get_bvn_derivs!(elbo_vars, gcc.bmc, x);
  gc = galaxy_prototypes[gcc_ind[3]][gcc_ind[2]]
  pc = mp.patches[s, b].psf[gcc_ind[1]]

  @test_approx_eq(
    pc.alphaBar * gc.etaBar * gcc.e_dev_i * exp(v) / (2 * pi),
    fs1m.v)

  @test_approx_eq fs1m.v f_wrap_gal(par)

  # Test the gradient.
  ad_grad_fun = ForwardDiff.gradient(f_wrap_gal);
  ad_grad = ad_grad_fun(par);
  @test_approx_eq ad_grad fs1m.d

  # Test the hessian.
  ad_hess_fun = ForwardDiff.hessian(f_wrap_gal)
  ad_hess = ad_hess_fun(par)
  @test_approx_eq_eps ad_hess fs1m.hs[1] 1e-10


  ###########################
  # Stars

  # Pick out a single star component for testing.
  # The index is psf, source
  bmc_ind = (1, s)
  function f_wrap_star{T <: Number}(par::Vector{T})
    # This uses mp, x, wcs_jacobian, and gcc_ind from the enclosing namespace.
    if T != Float64
      mp_fd = CelesteTypes.forward_diff_model_params(T, mp);
    else
      mp_fd = deepcopy(mp)
    end
    fs0m = zero_sensitive_float(StarPosParams, T, 1);

    # Make sure par is as long as the galaxy parameters.
    @assert length(par) == length(ids.u)
    for p1 in 1:2
        p0 = ids.u[p1]
        mp_fd.vp[s][p0] = par[p1]
    end
    star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(mp_fd, b);
    elbo_vars_fd = ElboDeriv.ElboIntermediateVariables(T);
    ElboDeriv.accum_star_pos!(
      elbo_vars_fd, star_mcs[bmc_ind...], x, fs0m, patch.wcs_jacobian);
    fs0m.v
  end

  function mp_to_par_star(mp::ModelParams{Float64})
    par = zeros(2)
    for p1 in 1:length(par)
        par[p1] = mp.vp[s][ids.u[p1]]
    end
    par
  end

  par = mp_to_par_star(mp)

  fs0m = zero_sensitive_float(StarPosParams, 1);
  star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(mp, b);
  ElboDeriv.accum_star_pos!(
    elbo_vars, star_mcs[bmc_ind...], x, fs0m, patch.wcs_jacobian);

  # One sanity check.
  @test_approx_eq fs0m.v f_wrap_star(par)

  # Test the gradient.
  ad_grad_fun = ForwardDiff.gradient(f_wrap_star);
  ad_grad = ad_grad_fun(par);
  @test_approx_eq ad_grad fs0m.d

  # Test the hessian.
  ad_hess_fun = ForwardDiff.hessian(f_wrap_star)
  ad_hess = ad_hess_fun(par)
  @test_approx_eq_eps ad_hess fs0m.hs[1] 1e-10
end


function test_galaxy_variable_transform()
  # TODO: test with a real and asymmetric wcs jacobian.
  # We only need this for a psf and jacobian.
  blob, mp, three_bodies = gen_three_body_dataset();

  # Pick a single source and band for testing.
  s = 1
  b = 3

  # The pixel and world centers shouldn't matter for derivatives.
  patch = mp.patches[s];
  psf = patch.psf[s];

  # Pick out a single galaxy component for testing.
  gp = galaxy_prototypes[1][1];
  e_dev_dir = 1.0;
  e_dev_i = 0.8;

  # Test the variable transformation.
  e_angle, e_axis, e_scale = (pi / 4, 0.7, 1.2)
  u = Float64[2.1, 3.1]
  x = Float64[2.8, 2.9]

  par_ids_u = [1, 2]
  par_ids_e_axis = 3
  par_ids_e_angle = 4
  par_ids_e_scale = 5
  par_ids_length = 5

  function wrap_par{T <: Number}(
      u::Vector{T}, e_angle::T, e_axis::T, e_scale::T)
    par = zeros(T, par_ids_length)
    par[par_ids_u] = u
    par[par_ids_e_angle] = e_angle
    par[par_ids_e_axis] = e_axis
    par[par_ids_e_scale] = e_scale
    par
  end

  function f_wrap{T <: Number}(par::Vector{T})
    u = par[par_ids_u]
    e_angle = par[par_ids_e_angle]
    e_axis = par[par_ids_e_axis]
    e_scale = par[par_ids_e_scale]
    u_pix = WCS.world_to_pixel(
      patch.wcs_jacobian, patch.center, patch.pixel_center, u)
    elbo_vars_fd = ElboDeriv.ElboIntermediateVariables(T)
    e_dev_i_fd = convert(T, e_dev_i)
    gcc = ElboDeriv.GalaxyCacheComponent(
            e_dev_dir, e_dev_i_fd, gp, psf, u_pix, e_axis, e_angle, e_scale);

    py1, py2, f_pre = ElboDeriv.eval_bvn_pdf(gcc.bmc, x);

    log(f_pre)
  end

  par = wrap_par(u, e_angle, e_axis, e_scale)
  u_pix = WCS.world_to_pixel(
    patch.wcs_jacobian, patch.center, patch.pixel_center, u)
  gcc = ElboDeriv.GalaxyCacheComponent(
          e_dev_dir, e_dev_i, gp, psf, u_pix, e_axis, e_angle, e_scale);
  elbo_vars = ElboDeriv.ElboIntermediateVariables(Float64)
  ElboDeriv.get_bvn_derivs!(elbo_vars, gcc.bmc, x);
  ElboDeriv.transform_bvn_derivs!(elbo_vars, gcc, patch.wcs_jacobian);

  # Sanity check the wrapper.
  @test_approx_eq(
    -0.5 *((x - gcc.bmc.the_mean)' * gcc.bmc.precision * (x - gcc.bmc.the_mean) -
           log(det(gcc.bmc.precision)))[1,1] - log(2pi) +
           log(psf.alphaBar * gp.etaBar),
    f_wrap(par))

  # Check the gradient.
  ad_grad_fun = ForwardDiff.gradient(f_wrap);
  ad_grad = ad_grad_fun(par)
  @test_approx_eq ad_grad [elbo_vars.bvn_u_d; elbo_vars.bvn_s_d]

  ad_hess_fun = ForwardDiff.hessian(f_wrap);
  ad_hess = ad_hess_fun(par);

  @test_approx_eq ad_hess[1:2, 1:2] elbo_vars.bvn_uu_h
  @test_approx_eq ad_hess[1:2, 3:5] elbo_vars.bvn_us_h

  # I'm not sure why this requires less precision for this test.
  @test_approx_eq_eps ad_hess[3:5, 3:5] elbo_vars.bvn_ss_h 1e-10
end


function test_galaxy_sigma_derivs()
  # Test d sigma / d shape

  e_angle, e_axis, e_scale = (pi / 4, 0.7, 1.2)

  function wrap_par{T <: Number}(e_angle::T, e_axis::T, e_scale::T)
    par = zeros(T, length(gal_shape_ids))
    par[gal_shape_ids.e_angle] = e_angle
    par[gal_shape_ids.e_axis] = e_axis
    par[gal_shape_ids.e_scale] = e_scale
    par
  end

  for si in 1:3
    sig_i = [(1, 1), (1, 2), (2, 2)][si]
    println("Testing sigma[$(sig_i)]")
    function f_wrap{T <: Number}(par::Vector{T})
      e_angle_fd = par[gal_shape_ids.e_angle]
      e_axis_fd = par[gal_shape_ids.e_axis]
      e_scale_fd = par[gal_shape_ids.e_scale]
      this_cov = Util.get_bvn_cov(e_axis_fd, e_angle_fd, e_scale_fd)
      this_cov[sig_i...]
    end

    par = wrap_par(e_angle, e_axis, e_scale)
    XiXi = Util.get_bvn_cov(e_axis, e_angle, e_scale)

    gal_derivs = ElboDeriv.GalaxySigmaDerivs(e_angle, e_axis, e_scale, XiXi);

    ad_grad_fun = ForwardDiff.gradient(f_wrap);
    ad_grad = ad_grad_fun(par);
    @test_approx_eq gal_derivs.j[si, :][:] ad_grad

    ad_hess_fun = ForwardDiff.hessian(f_wrap);
    ad_hess = ad_hess_fun(par);
    @test_approx_eq(
      ad_hess,
      reshape(gal_derivs.t[si, :, :],
              length(gal_shape_ids), length(gal_shape_ids)))
  end
end


function test_bvn_derivatives()
  # Test log(bvn prob) / d(mean, sigma)

  x = Float64[2.0, 3.0]
  sigma = Float64[1.0 0.2; 0.2 1.0]
  offset = Float64[0.5, 0.5]

  # Note that get_bvn_derivs doesn't use the weight, so set it to something
  # strange to check that it doesn't matter.
  weight = 0.724

  bvn = ElboDeriv.BvnComponent(offset, sigma, weight);
  elbo_vars = ElboDeriv.ElboIntermediateVariables(Float64);
  v = ElboDeriv.get_bvn_derivs!(elbo_vars, bvn, x);

  function f{T <: Number}(x::Vector{T}, sigma::Matrix{T})
    local_x = x - offset
    -0.5 * ((local_x' * (sigma \ local_x))[1,1] + log(det(sigma)))
  end

  x_ids = 1:2
  sig_ids = 3:5
  function wrap(x::Vector{Float64}, sigma::Matrix{Float64})
    par = zeros(Float64, length(x_ids) + length(sig_ids))
    par[x_ids] = x
    par[sig_ids] = [ sigma[1, 1], sigma[1, 2], sigma[2, 2]]
    par
  end

  function f_wrap{T <: Number}(par::Vector{T})
    x_loc = par[x_ids]
    s_vec = par[sig_ids]
    sig_loc = T[s_vec[1] s_vec[2]; s_vec[2] s_vec[3]]
    f(x_loc, sig_loc)
  end

  par = wrap(x, sigma);
  @test_approx_eq v f_wrap(par)

  ad_grad_fun = ForwardDiff.gradient(f_wrap);
  ad_d = ad_grad_fun(par);
  @test_approx_eq elbo_vars.bvn_x_d ad_d[x_ids]
  @test_approx_eq elbo_vars.bvn_sig_d ad_d[sig_ids]

  ad_hess_fun = ForwardDiff.hessian(f_wrap);
  ad_h = ad_hess_fun(par);
  @test_approx_eq elbo_vars.bvn_xx_h ad_h[x_ids, x_ids]
  @test_approx_eq elbo_vars.bvn_xsig_h ad_h[x_ids, sig_ids]
  @test_approx_eq elbo_vars.bvn_sigsig_h ad_h[sig_ids, sig_ids]
end


function test_brightness_hessian()
    blob, mp, three_bodies = gen_three_body_dataset();
    kept_ids = [ ids.r1; ids.r2; ids.c1[:]; ids.c2[:] ];
    omitted_ids = setdiff(1:length(ids), kept_ids);

    transform = Transform.get_identity_transform(length(ids), mp.S);
    ad_hess = zeros(Float64, length(kept_ids) * mp.S, length(kept_ids) * mp.S);
    for i = 1:Ia, b = [3,4,2,5,1], squares in [false, true]
        # Write the brightness as an ordinary function of a ModelParams
        # so we can use the objective function autodiff hessian logic.
        function wrap_source_brightness{NumType <: Number}(
            mp::ModelParams{NumType})
          ret = zero_sensitive_float(CanonicalParams, NumType, mp.S);
          for s=1:mp.S
            sb = ElboDeriv.SourceBrightness(mp.vp[s]);
            if squares
              ret.v = sb.E_l_a[b, i].v
              ret.d[:, s] = sb.E_l_a[b, i].d
              ret.hs[s][:, :] = sb.E_l_a[b, i].hs[1][:, :]
            else
              ret.v = sb.E_ll_a[b, i].v
              ret.d[:, s] = sb.E_ll_a[b, i].d
              ret.hs[s][:, :] = sb.E_ll_a[b, i].hs[1][:, :]
            end
          end
          ret
        end

        bright = wrap_source_brightness(mp);
        objective = OptimizeElbo.ObjectiveWrapperFunctions(
          wrap_source_brightness, mp, transform, kept_ids, omitted_ids);
        x = transform.vp_to_array(mp.vp, omitted_ids);

        # Sanity check.
        @test_approx_eq objective.f_value(x[:]) bright.v

        # Compare the AD hessian with the exact hessian.
        objective.f_ad_hessian!(x[:], ad_hess);
        for s=1:mp.S
          hess_ind = (1:length(kept_ids)) + (s - 1) * length(kept_ids)
          hess0 = ad_hess[hess_ind, hess_ind]
          hess1 = bright.hs[s][kept_ids, kept_ids]
          @test_approx_eq(
            ad_hess[hess_ind, hess_ind], bright.hs[s][kept_ids, kept_ids])
        end
    end
end


function test_multiply_sf()
  # Test for hessians.
  # Two sets of ids with some overlap and some disjointness.
  p = length(ids)
  S = 2

  ids1 = find((1:p) .% 2 .== 0)
  ids2 = setdiff(1:p, ids1)
  ids1 = union(ids1, 1:5)
  ids2 = union(ids2, 1:5)

  l1 = zeros(Float64, S * p);
  l2 = zeros(Float64, S * p);
  l1[ids1] = rand(length(ids1))
  l2[ids2] = rand(length(ids2))
  l1[ids1 + p] = rand(length(ids1))
  l2[ids2 + p] = rand(length(ids2))

  sigma1 = zeros(Float64, S * p, S * p);
  sigma2 = zeros(Float64, S * p, S * p);
  sigma1[ids1, ids1] = rand(length(ids1), length(ids1));
  sigma2[ids2, ids2] = rand(length(ids2), length(ids2));
  sigma1[ids1 + p, ids1 + p] = rand(length(ids1), length(ids1));
  sigma2[ids2 + p, ids2 + p] = rand(length(ids2), length(ids2));
  sigma1 = 0.5 * (sigma1 + sigma1')
  sigma2 = 0.5 * (sigma2 + sigma2')

  x = 0.1 * rand(S * p);

  function testfun1(x)
    (l1' * x + 0.5 * x' * sigma1 * x)[1,1]
  end

  function testfun2(x)
    (l2' * x + 0.5 * x' * sigma2 * x)[1,1]
  end

  function testfun(x)
    testfun1(x) * testfun2(x)
  end

  ret1 = zero_sensitive_float(CanonicalParams, Float64, S);
  ret2 = zero_sensitive_float(CanonicalParams, Float64, S);
  s_ind = Array(UnitRange{Int64}, 2);
  s_ind[1] = 1:p
  s_ind[2] = (1:p) + p

  ret1.v = testfun1(x)
  fill!(ret1.d, 0.0);
  for s=1:S
    fill!(ret1.hs[s], 0.0);
    ret1.d[:, s] = l1[s_ind[s]] + sigma1[s_ind[s], s_ind[s]] * x[s_ind[s]];
    ret1.hs[s] = sigma1[s_ind[s], s_ind[s]];
  end

  ret2.v = testfun2(x)
  fill!(ret2.d, 0.0);
  for s=1:S
    fill!(ret2.hs[s], 0.0);
    ret2.d[:, s] = l2[s_ind[s]] + sigma2[s_ind[s], s_ind[s]] * x[s_ind[s]];
    ret2.hs[s] = sigma2[s_ind[s], s_ind[s]];
  end

  hess = zeros(Float64, S * p, S * p);
  grad = ForwardDiff.gradient(testfun1, x);
  ForwardDiff.hessian!(hess, testfun1, x);
  for s=1:S
    @test_approx_eq(ret1.d[:, s], grad[s_ind[s]])
    @test_approx_eq(ret1.hs[s], hess[s_ind[s], s_ind[s]])
  end

  grad = ForwardDiff.gradient(testfun2, x);
  ForwardDiff.hessian!(hess, testfun2, x);
  for s=1:S
    @test_approx_eq(ret2.d[:, s], grad[s_ind[s]])
    @test_approx_eq(ret2.hs[s], hess[s_ind[s], s_ind[s]])
  end


  grad = ForwardDiff.gradient(testfun, x);
  ForwardDiff.hessian!(hess, testfun, x);

  sf1 = deepcopy(ret1);
  sf2 = deepcopy(ret2);
  multiply_sf!(sf1, sf2, ids1=ids1, ids2=ids2);

  for s=1:S
    @test_approx_eq(sf1.d[:, s], grad[s_ind[s]])
    @test_approx_eq(sf1.hs[s][:], hess[s_ind[s], s_ind[s]])
  end
end


function test_set_hess()
  sf = zero_sensitive_float(CanonicalParams, 2);
  CelesteTypes.set_hess!(sf, 2, 3, 5.0);
  @test_approx_eq sf.hs[1][2, 3] 5.0
  @test_approx_eq sf.hs[1][3, 2] 5.0

  CelesteTypes.set_hess!(sf, 4, 4, 6.0);
  @test_approx_eq sf.hs[1][4, 4] 6.0

  CelesteTypes.set_hess!(sf, 2, 3, 2, 7.0);
  @test_approx_eq sf.hs[2][2, 3] 7.0
  @test_approx_eq sf.hs[2][3, 2] 7.0

  CelesteTypes.set_hess!(sf, 4, 4, 2, 8.0);
  @test_approx_eq sf.hs[2][4, 4] 8.0
end


test_set_hess()
test_multiply_sf()
test_brightness_hessian()
test_bvn_derivatives()
test_galaxy_sigma_derivs()
test_galaxy_variable_transform()
test_fsXm_derivatives()
