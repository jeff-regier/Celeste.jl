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

###########################
# Galaxies

# Pick out a single galaxy component for testing.
# The index is (psf, galaxy, gal type, source)
#for psf_k=1:3, type_i = 1:2, gal_j in 1:[8,6][type_i]
  #psf_k = 1; type_i = 1; gal_j = 1
  #gcc_ind = (psf_k, gal_j, type_i, s)
  gcc_ind = (1, 4, 2, s)
  println(gcc_ind)
  function f_wrap_gal{T <: Number}(par::Vector{T})
    # This uses mp, x, wcs_jacobian, and gcc_ind from the enclosing namespace.
    if T != Float64
      mp_fd = CelesteTypes.forward_diff_model_params(T, mp);
    else
      mp_fd = deepcopy(mp);
    end

    # Make sure par is as long as the galaxy parameters.
    @assert length(par) == length(shape_standard_alignment[2])
    for p1 in 1:length(par)
        p0 = shape_standard_alignment[2][p1]
        mp_fd.vp[s][p0] = par[p1]
    end
    star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(mp_fd, b);

    # Raw:
    gcc = gal_mcs[gcc_ind...];
    py1, py2, f_pre = ElboDeriv.eval_bvn_pdf(gcc.bmc, x)
    f_pre * gcc.e_dev_i

    # Through accum_galaxy_pos!
    # elbo_vars_fd = ElboDeriv.ElboIntermediateVariables(T, 1);
    # ElboDeriv.accum_galaxy_pos!(
    #   elbo_vars_fd, s, gal_mcs[gcc_ind...], x, patch.wcs_jacobian);
    # elbo_vars_fd.fs1m_vec[s].v
  end

  function mp_to_par_gal(mp::ModelParams{Float64})
    par = zeros(length(shape_standard_alignment[2]))
    for p1 in 1:length(par)
        p0 = shape_standard_alignment[2][p1]
        par[p1] = mp.vp[s][p0]
    end
    par
  end

  par_gal = mp_to_par_gal(mp);

  star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(mp, b);
  clear!(elbo_vars.fs1m_vec[s]);
  ElboDeriv.accum_galaxy_pos!(
    elbo_vars, s, gal_mcs[gcc_ind...], x, patch.wcs_jacobian);
  fs1m = deepcopy(elbo_vars.fs1m_vec[s]);

  # Two sanity checks.
  gcc = gal_mcs[gcc_ind...];
  clear!(elbo_vars.fs1m_vec[s]);
  v = ElboDeriv.get_bvn_derivs!(elbo_vars, gcc.bmc, x);
  gc = galaxy_prototypes[gcc_ind[3]][gcc_ind[2]]
  pc = mp.patches[s, b].psf[gcc_ind[1]]

  @test_approx_eq(
    pc.alphaBar * gc.etaBar * gcc.e_dev_i * exp(v) / (2 * pi),
    fs1m.v)

  @test_approx_eq fs1m.v f_wrap_gal(par_gal)

  # Test the gradient.
  ad_grad_gal = ForwardDiff.gradient(f_wrap_gal, par_gal);
  @test_approx_eq ad_grad_gal fs1m.d

  # Test the hessian.
  # Currently broken!
  ad_hess_gal = ForwardDiff.hessian(f_wrap_gal, par_gal)
  #@test_approx_eq_eps ad_hess_gal fs1m.h 1e-6
  println(maximum(abs(ad_hess_gal - fs1m.h)))
  ad_hess_gal - fs1m.h
#end
















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

###########################
# Galaxies

# Pick out a single galaxy component for testing.
# The index is psf, galaxy, gal type, source
for psf_k=1:3, type_i = 1:2, gal_j in 1:[8,6][type_i]
  gcc_ind = (psf_k, gal_j, type_i, s)

  #gcc_ind = (1, 5, 2, 1) # This one is really bad.
  println(gcc_ind)
  #gcc_ind = (1, 1, 1, s)
  function f_wrap_gal{T <: Number}(par::Vector{T})
    # This uses mp, x, wcs_jacobian, and gcc_ind from the enclosing namespace.
    if T != Float64
      mp_fd = CelesteTypes.forward_diff_model_params(T, mp);
    else
      mp_fd = deepcopy(mp)
    end

    # Make sure par is as long as the galaxy parameters.
    @assert length(par) == length(shape_standard_alignment[2])
    for p1 in 1:length(par)
        p0 = shape_standard_alignment[2][p1]
        mp_fd.vp[s][p0] = par[p1]
    end
    star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(mp_fd, b);
    elbo_vars_fd = ElboDeriv.ElboIntermediateVariables(T, 1);
    ElboDeriv.accum_galaxy_pos!(
      elbo_vars_fd, s, gal_mcs[gcc_ind...], x, patch.wcs_jacobian);
    elbo_vars_fd.fs1m_vec[s].v
  end

  function mp_to_par_gal(mp::ModelParams{Float64})
    par = zeros(length(shape_standard_alignment[2]))
    for p1 in 1:length(par)
        p0 = shape_standard_alignment[2][p1]
        par[p1] = mp.vp[s][p0]
    end
    par
  end

  par_gal = mp_to_par_gal(mp);

  star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(mp, b);
  clear!(elbo_vars.fs1m_vec[s]);
  ElboDeriv.accum_galaxy_pos!(
    elbo_vars, s, gal_mcs[gcc_ind...], x, patch.wcs_jacobian);
  fs1m = deepcopy(elbo_vars.fs1m_vec[s]);

  # Two sanity checks.
  gcc = gal_mcs[gcc_ind...];
  println(gcc.e_dev_dir)
  clear!(elbo_vars.fs1m_vec[s]);
  v = ElboDeriv.get_bvn_derivs!(elbo_vars, gcc.bmc, x);
  gc = galaxy_prototypes[gcc_ind[3]][gcc_ind[2]]
  pc = mp.patches[s, b].psf[gcc_ind[1]]

  @test_approx_eq(
    pc.alphaBar * gc.etaBar * gcc.e_dev_i * exp(v) / (2 * pi),
    fs1m.v)

  @test_approx_eq fs1m.v f_wrap_gal(par_gal)

  # Test the gradient.
  ad_grad_gal = ForwardDiff.gradient(f_wrap_gal, par_gal);
  @test_approx_eq ad_grad_gal fs1m.d

  # Test the hessian.
  # Currently broken!
  ad_hess_gal = ForwardDiff.hessian(f_wrap_gal, par_gal)
  #@test_approx_eq_eps ad_hess_gal fs1m.h 1e-6
  println(maximum(abs(ad_hess_gal - fs1m.h)))
end



function test_galaxy_variable_transform()
  # TODO: this is not passing with certain components.

  # TODO: test with a real and asymmetric wcs jacobian.
  # We only need this for a psf and jacobian.
  blob, mp, three_bodies = gen_three_body_dataset();

  # Pick a single source and band for testing.
  s = 1
  b = 3

  # The pixel and world centers shouldn't matter for derivatives.
  patch = mp.patches[s, b];
  psf = patch.psf[1];

  # Pick out a single galaxy component for testing.
  gp = galaxy_prototypes[4][2];
  e_dev_dir = -1.0;
  e_dev_i = 0.85;

  # Test the variable transformation.
  e_angle, e_axis, e_scale = (1.1, 0.75, 4.8)
  u = Float64[5.3, 2.9]
  x = Float64[7.0, 5.0]

  # The indices in par of each variable.
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
    elbo_vars_fd = ElboDeriv.ElboIntermediateVariables(T, 1)
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
  elbo_vars = ElboDeriv.ElboIntermediateVariables(Float64, 1);
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
  ad_hess[3:5, 3:5] - elbo_vars.bvn_ss_h

end





















function test_e_g_functions()
  # Currently broken due to broken fs1m derivatives.

  blob, mp, bodies, tiled_blob = gen_sample_galaxy_dataset();

  S = length(mp.active_sources)
  P = length(CanonicalParams)
  b = 3
  h = 10
  w = 10
  tile = tiled_blob[b][1,1]; # Note: only one tile in this simulated dataset.

  # For debugging.  Only the galaxy hessian is wrong for E_G.
  mp.vp[1][ids.a] = [0. 1.]

  test_squares = false

  function e_g_wrapper_fun{NumType <: Number}(mp::ModelParams{NumType})
    star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(mp, b);
    sbs = ElboDeriv.SourceBrightness{NumType}[
      ElboDeriv.SourceBrightness(mp.vp[s]) for s in 1:mp.S];

    elbo_vars_loc = ElboDeriv.ElboIntermediateVariables(NumType, mp.S);
    ElboDeriv.populate_fsm_vecs!(
      elbo_vars_loc, mp, tile, h, w, sbs, gal_mcs, star_mcs);

    E_G = elbo_vars_loc.E_G;
    E_G2 = elbo_vars_loc.E_G2;

    clear!(E_G);
    clear!(E_G2);

    ElboDeriv.combine_pixel_sources!(elbo_vars_loc, mp, tile, sbs);
    elbo_vars_loc
  end

  function wrapper_fun{NumType <: Number}(x::Vector{NumType})
    @assert length(x) == S * P
    x_mat = reshape(x, (P, S))
    if NumType != Float64
      mp_fd = CelesteTypes.forward_diff_model_params(NumType, mp);
    else
      mp_fd = deepcopy(mp)
    end
    for sa_ind in 1:length(mp_fd.active_sources)
      mp_fd.vp[mp.active_sources[sa_ind]] = x_mat[:, sa_ind]
    end
    elbo_vars_fd = e_g_wrapper_fun(mp_fd)
    test_squares ? elbo_vars_fd.E_G2.v : elbo_vars_fd.E_G.v
  end


  x_mat = zeros(Float64, P, S);
  for sa_ind in 1:S
    x_mat[:, sa_ind] = mp.vp[mp.active_sources[sa_ind]]
  end
  x = x_mat[:];

  elbo_vars = e_g_wrapper_fun(mp);
  sf = test_squares ? deepcopy(elbo_vars.E_G2) : deepcopy(elbo_vars.E_G);

  v = wrapper_fun(x)
  @test_approx_eq v sf.v

  grad = ForwardDiff.gradient(wrapper_fun, x);
  @test_approx_eq grad sf.d

  hess = ForwardDiff.hessian(wrapper_fun, x);
  #@test_approx_eq hess elbo_vars.E_G.h
  matshow(abs(sf.h - hess) .> 1e-6)

end


function test_combine_sfs()
  # TODO: this test was designed for multiply_sf.  Make it more general.

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
  sigma1 = 0.5 * (sigma1 + sigma1');
  sigma2 = 0.5 * (sigma2 + sigma2');

  x = 0.1 * rand(S * p);

  function base_fun1{T <: Number}(x::Vector{T})
    (l1' * x + 0.5 * x' * sigma1 * x)[1,1]
  end

  function base_fun2{T <: Number}(x::Vector{T})
    (l2' * x + 0.5 * x' * sigma2 * x)[1,1]
  end

  function multiply_fun{T <: Number}(x::Vector{T})
    base_fun1(x) * base_fun1(x)
  end

  function combine_fun{T <: Number}(x::Vector{T})
    (base_fun1(x) ^ 2) * sqrt(base_fun2(x))
  end

  function combine_fun_derivatives{T <: Number}(x::Vector{T})
    g_d = T[2 * base_fun1(x) * sqrt(base_fun2(x)),
            0.5 * (base_fun1(x) ^ 2) / sqrt(base_fun2(x)) ]
    g_h = zeros(T, 2, 2)
    g_h[1, 1] = 2 * sqrt(base_fun2(x))
    g_h[2, 2] = -0.25 * (base_fun1(x) ^ 2) * (base_fun2(x) ^(-3/2))
    g_h[1, 2] = g_h[2, 1] = base_fun1(x) / sqrt(base_fun2(x))
    g_d, g_h
  end


  s_ind = Array(UnitRange{Int64}, 2);
  s_ind[1] = 1:p
  s_ind[2] = (1:p) + p

  ret1 = zero_sensitive_float(CanonicalParams, Float64, S);
  ret1.v = base_fun1(x)
  fill!(ret1.d, 0.0);
  fill!(ret1.h, 0.0);
  for s=1:S
    ret1.d[:, s] = l1[s_ind[s]] + sigma1[s_ind[s], s_ind[s]] * x[s_ind[s]];
    ret1.h[s_ind[s], s_ind[s]] = sigma1[s_ind[s], s_ind[s]];
  end

  ret2 = zero_sensitive_float(CanonicalParams, Float64, S);
  ret2.v = base_fun2(x)
  fill!(ret2.d, 0.0);
  fill!(ret2.h, 0.0);
  for s=1:S
    ret2.d[:, s] = l2[s_ind[s]] + sigma2[s_ind[s], s_ind[s]] * x[s_ind[s]];
    ret2.h[s_ind[s], s_ind[s]] = sigma2[s_ind[s], s_ind[s]];
  end

  grad = ForwardDiff.gradient(base_fun1, x);
  hess = ForwardDiff.hessian(base_fun1, x);
  for s=1:S
    @test_approx_eq(ret1.d[:, s], grad[s_ind[s]])
  end
  @test_approx_eq(ret1.h, hess)

  grad = ForwardDiff.gradient(base_fun2, x);
  hess = ForwardDiff.hessian(base_fun2, x);
  for s=1:S
    @test_approx_eq(ret2.d[:, s], grad[s_ind[s]])
  end
  @test_approx_eq(ret2.h, hess)

  # Test the combinations.
  v = combine_fun(x);
  grad = ForwardDiff.gradient(combine_fun, x);
  hess = ForwardDiff.hessian(combine_fun, x);

  sf1 = deepcopy(ret1);
  sf2 = deepcopy(ret2);
  g_d, g_h = combine_fun_derivatives(x)
  CelesteTypes.combine_sfs!(sf1, sf2, sf1.v ^ 2 * sqrt(sf2.v), g_d, g_h);

  @test_approx_eq sf1.v v
  @test_approx_eq sf1.d[:] grad
  @test_approx_eq sf1.h hess
end



function test_fs0m_derivatives()
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

    # Make sure par is as long as the galaxy parameters.
    @assert length(par) == length(ids.u)
    for p1 in 1:2
        p0 = ids.u[p1]
        mp_fd.vp[s][p0] = par[p1]
    end
    star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(mp_fd, b);
    elbo_vars_fd = ElboDeriv.ElboIntermediateVariables(T, 1);
    ElboDeriv.accum_star_pos!(
      elbo_vars_fd, s, star_mcs[bmc_ind...], x, patch.wcs_jacobian);
    elbo_vars_fd.fs0m_vec[s].v
  end

  function mp_to_par_star(mp::ModelParams{Float64})
    par = zeros(2)
    for p1 in 1:length(par)
        par[p1] = mp.vp[s][ids.u[p1]]
    end
    par
  end

  par_star = mp_to_par_star(mp)

  clear!(elbo_vars.fs0m_vec[s])
  star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(mp, b);
  ElboDeriv.accum_star_pos!(
    elbo_vars, s, star_mcs[bmc_ind...], x, patch.wcs_jacobian);
  fs0m = deepcopy(elbo_vars.fs0m_vec[s])

  # One sanity check.
  @test_approx_eq fs0m.v f_wrap_star(par_star)

  # Test the gradient.
  ad_grad_star = ForwardDiff.gradient(f_wrap_star, par_star);
  @test_approx_eq ad_grad_star fs0m.d

  # Test the hessian.
  ad_hess_star = ForwardDiff.hessian(f_wrap_star, par_star)
  @test_approx_eq_eps ad_hess_star fs0m.h 1e-10
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
  elbo_vars = ElboDeriv.ElboIntermediateVariables(Float64, 1);
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
  blob, mp, star_cat = gen_sample_star_dataset();
  kept_ids = [ ids.r1; ids.r2; ids.c1[:]; ids.c2[:] ];
  omitted_ids = setdiff(1:length(ids), kept_ids);
  b = 3
  i = 1

  for squares in [false, true]
    function wrap_source_brightness{NumType <: Number}(
        vp::Vector{NumType})
      ret = zero_sensitive_float(CanonicalParams, NumType);
      sb = ElboDeriv.SourceBrightness(vp);
      if squares
        ret.v = sb.E_ll_a[b, i].v;
        ret.d[:, 1] = sb.E_ll_a[b, i].d;
        ret.h[:, :] = sb.E_ll_a[b, i].h[:, :];
      else
        ret.v = sb.E_l_a[b, i].v;
        ret.d[:, 1] = sb.E_l_a[b, i].d;
        ret.h[:, :] = sb.E_l_a[b, i].h[:, :];
      end
      ret
    end

    function wrap_source_brightness_value{NumType <: Number}(
        vp::Vector{NumType})
      wrap_source_brightness(vp).v
    end

    bright = wrap_source_brightness(mp.vp[1]);

    ad_grad_fun = ForwardDiff.gradient(wrap_source_brightness_value);
    ad_grad = ad_grad_fun(mp.vp[1]);
    @test_approx_eq ad_grad bright.d[:, 1]

    ad_hess_fun = ForwardDiff.hessian(wrap_source_brightness_value);
    ad_hess = ad_hess_fun(mp.vp[1]);
    @test_approx_eq ad_hess bright.h
  end
end


function test_set_hess()
  sf = zero_sensitive_float(CanonicalParams);
  CelesteTypes.set_hess!(sf, 2, 3, 5.0);
  @test_approx_eq sf.h[2, 3] 5.0
  @test_approx_eq sf.h[3, 2] 5.0

  CelesteTypes.set_hess!(sf, 4, 4, 6.0);
  @test_approx_eq sf.h[4, 4] 6.0
end


test_combine_sfs()
test_set_hess()
test_brightness_hessian()
test_bvn_derivatives()
test_galaxy_sigma_derivs()
test_galaxy_variable_transform()
test_fsXm_derivatives()
