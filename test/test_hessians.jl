using Celeste
using CelesteTypes
using Base.Test
# using Distributions
using SampleData
# using Transform
# using PyPlot

# using ForwardDiff

import Synthetic
# import WCS


println("Running hessian tests.")


function test_tile_likelihood()
  blob, mp, bodies, tiled_blob = gen_two_body_dataset();
  b = 1
  # Only keep a few pixels to make the autodiff results faster.
  keep_pixels = 10:11
  tile = tiled_blob[b][1, 1];
  tile.pixels[setdiff(1:tile.h_width, keep_pixels), :] = NaN;
  tile.pixels[:, setdiff(1:tile.w_width, keep_pixels)] = NaN;

  function tile_lik_wrapper_fun{NumType <: Number}(
      mp::ModelParams{NumType}, calculate_derivs::Bool)

    elbo_vars = ElboDeriv.ElboIntermediateVariables(NumType, mp.S, mp.S);
    elbo_vars.calculate_derivs = calculate_derivs
    star_mcs, gal_mcs =
      ElboDeriv.load_bvn_mixtures(mp, b, calculate_derivs=elbo_vars.calculate_derivs);
    sbs = ElboDeriv.load_source_brightnesses(mp, elbo_vars.calculate_derivs);
    ElboDeriv.tile_likelihood!(elbo_vars, tile, mp, sbs, star_mcs, gal_mcs);
    deepcopy(elbo_vars.elbo)
  end

  function tile_lik_value_wrapper{NumType <: Number}(x::Vector{NumType})
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
    tile_lik_wrapper_fun(mp_fd, false).v
  end

  elbo = tile_lik_wrapper_fun(mp, true);

  P = length(ids)
  S = mp.S
  x_mat = zeros(Float64, P, S);
  for sa_ind in 1:S
    x_mat[:, sa_ind] = mp.vp[mp.active_sources[sa_ind]]
  end
  x = x_mat[:];

  @test_approx_eq tile_lik_value_wrapper(x) elbo.v

  ad_grad = ForwardDiff.gradient(tile_lik_value_wrapper, x);
  @test_approx_eq ad_grad elbo.d

  ad_hess = ForwardDiff.hessian(tile_lik_value_wrapper, x);
  @test_approx_eq ad_hess elbo.h
end


function test_add_log_term()
  blob, mp, bodies, tiled_blob = gen_two_body_dataset();

  for b = 1:5
    println("Testing log term for band $b.")
    x_nbm = 70.
    iota = blob[b].iota

    function add_log_term_wrapper_fun{NumType <: Number}(
        mp::ModelParams{NumType}, calculate_derivs::Bool)

      star_mcs, gal_mcs =
        ElboDeriv.load_bvn_mixtures(mp, b, calculate_derivs=calculate_derivs);
      sbs = ElboDeriv.SourceBrightness{NumType}[
        ElboDeriv.SourceBrightness(mp.vp[s], calculate_derivs=calculate_derivs)
        for s in 1:mp.S];

      elbo_vars_loc = ElboDeriv.ElboIntermediateVariables(NumType, mp.S, mp.S);
      elbo_vars_loc.calculate_derivs = calculate_derivs
      ElboDeriv.populate_fsm_vecs!(
        elbo_vars_loc, mp, tile, h, w, sbs, gal_mcs, star_mcs);
      ElboDeriv.combine_pixel_sources!(elbo_vars_loc, mp, tile, sbs);

      ElboDeriv.add_elbo_log_term!(elbo_vars_loc, x_nbm, iota)

      deepcopy(elbo_vars_loc.elbo)
    end

    function ad_wrapper_fun{NumType <: Number}(x::Vector{NumType})
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
      add_log_term_wrapper_fun(mp_fd, false).v
    end


    P = length(ids)
    S = mp.S
    x_mat = zeros(Float64, P, S);
    for sa_ind in 1:S
      x_mat[:, sa_ind] = mp.vp[mp.active_sources[sa_ind]]
    end
    x = x_mat[:];

    elbo = add_log_term_wrapper_fun(mp, true);

    @test_approx_eq elbo.v ad_wrapper_fun(x)

    ad_grad = ForwardDiff.gradient(ad_wrapper_fun, x);
    @test_approx_eq ad_grad elbo.d[:]

    ad_hess = ForwardDiff.hessian(ad_wrapper_fun, x);
    @test_approx_eq ad_hess elbo.h[:]
  end
end


function test_combine_pixel_sources()
  blob, mp, bodies, tiled_blob = gen_two_body_dataset();

  S = length(mp.active_sources)
  P = length(CanonicalParams)
  h = 10
  w = 10

  for test_var = [false, true], b=1:5
    test_var_string = test_var ? "E_G" : "var_G"
    println("Testing $(test_var_string), band $b")

    tile = tiled_blob[b][1,1]; # Note: only one tile in this simulated dataset.
    function e_g_wrapper_fun{NumType <: Number}(
        mp::ModelParams{NumType}; calculate_derivs=true)

      star_mcs, gal_mcs =
        ElboDeriv.load_bvn_mixtures(mp, b, calculate_derivs=calculate_derivs);
      sbs = ElboDeriv.SourceBrightness{NumType}[
        ElboDeriv.SourceBrightness(mp.vp[s], calculate_derivs=calculate_derivs)
        for s in 1:mp.S];

      elbo_vars_loc = ElboDeriv.ElboIntermediateVariables(NumType, mp.S, mp.S);
      elbo_vars_loc.calculate_derivs = calculate_derivs;
      ElboDeriv.populate_fsm_vecs!(
        elbo_vars_loc, mp, tile, h, w, sbs, gal_mcs, star_mcs);
      ElboDeriv.combine_pixel_sources!(elbo_vars_loc, mp, tile, sbs);
      deepcopy(elbo_vars_loc)
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
      elbo_vars_fd = e_g_wrapper_fun(mp_fd, calculate_derivs=false)
      test_var ? elbo_vars_fd.var_G.v : elbo_vars_fd.E_G.v
    end

    x_mat = zeros(Float64, P, S);
    for sa_ind in 1:S
      x_mat[:, sa_ind] = mp.vp[mp.active_sources[sa_ind]]
    end
    x = x_mat[:];

    elbo_vars = e_g_wrapper_fun(mp);
    sf = test_var ? deepcopy(elbo_vars.var_G) : deepcopy(elbo_vars.E_G);

    v = wrapper_fun(x)
    @test_approx_eq v sf.v

    ad_grad = ForwardDiff.gradient(wrapper_fun, x);
    @test_approx_eq ad_grad sf.d

    ad_hess = ForwardDiff.hessian(wrapper_fun, x);
    @test_approx_eq ad_hess sf.h

  end
end


function test_e_g_s_functions()
  blob, mp, bodies, tiled_blob = gen_two_body_dataset();

  S = length(mp.active_sources)
  P = length(CanonicalParams)
  h = 10
  w = 10
  s = 1

  test_var = false
  b = 1

  for test_var = [false, true], b=1:5
    test_var_string = test_var ? "E_G" : "var_G"
    println("Testing $(test_var_string), band $b")

    tile = tiled_blob[b][1,1]; # Note: only one tile in this simulated dataset.
    function e_g_wrapper_fun{NumType <: Number}(
        mp::ModelParams{NumType}; calculate_derivs=true)

      star_mcs, gal_mcs =
        ElboDeriv.load_bvn_mixtures(mp, b, calculate_derivs=calculate_derivs);
      sbs = ElboDeriv.SourceBrightness{NumType}[
        ElboDeriv.SourceBrightness(mp.vp[s], calculate_derivs=calculate_derivs)
        for s in 1:mp.S];

      elbo_vars_loc = ElboDeriv.ElboIntermediateVariables(NumType, mp.S, mp.S);
      elbo_vars_loc.calculate_derivs = calculate_derivs;
      ElboDeriv.populate_fsm_vecs!(
        elbo_vars_loc, mp, tile, h, w, sbs, gal_mcs, star_mcs);
      ElboDeriv.accumulate_source_brightness!(elbo_vars_loc, mp, sbs, s, b);
      deepcopy(elbo_vars_loc)
    end

    function wrapper_fun{NumType <: Number}(x::Vector{NumType})
      @assert length(x) == P
      if NumType != Float64
        mp_fd = CelesteTypes.forward_diff_model_params(NumType, mp);
      else
        mp_fd = deepcopy(mp)
      end
      mp_fd.vp[s] = x
      elbo_vars_fd = e_g_wrapper_fun(mp_fd, calculate_derivs=false)
      test_var ? elbo_vars_fd.var_G_s.v : elbo_vars_fd.E_G_s.v
    end

    x = mp.vp[s];

    elbo_vars = e_g_wrapper_fun(mp);

    # Sanity check
    @test_approx_eq(elbo_vars.var_G_s.v,
                    elbo_vars.E_G2_s.v - (elbo_vars.E_G_s.v ^ 2))

    sf = test_var ? deepcopy(elbo_vars.var_G_s) : deepcopy(elbo_vars.E_G_s);

    v = wrapper_fun(x)
    @test_approx_eq v sf.v

    ad_grad = ForwardDiff.gradient(wrapper_fun, x);
    @test_approx_eq ad_grad sf.d

    ad_hess = ForwardDiff.hessian(wrapper_fun, x);
    @test_approx_eq ad_hess sf.h
  end
end


function test_fs1m_derivatives()
  # TODO: test with a real and asymmetric wcs jacobian.
  blob, mp, three_bodies = gen_three_body_dataset();
  omitted_ids = Int64[];
  kept_ids = setdiff(1:length(ids), omitted_ids);

  s = 1
  b = 1

  patch = mp.patches[s, b];
  u = mp.vp[s][ids.u]
  u_pix = WCS.world_to_pixel(
    patch.wcs_jacobian, patch.center, patch.pixel_center, u)
  x = ceil(u_pix + [1.0, 2.0])

  elbo_vars = ElboDeriv.ElboIntermediateVariables(Float64, 1, 1);

  ###########################
  # Galaxies

  # Pick out a single galaxy component for testing.
  # The index is (psf, galaxy, gal type, source)
  for psf_k=1:3, type_i = 1:2, gal_j in 1:[8,6][type_i]
    #psf_k = 1; type_i = 1; gal_j = 2
    gcc_ind = (psf_k, gal_j, type_i, s)
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
      star_mcs, gal_mcs =
        ElboDeriv.load_bvn_mixtures(mp_fd, b, calculate_derivs=false);

      # Raw:
      gcc = gal_mcs[gcc_ind...];
      py1, py2, f_pre = ElboDeriv.eval_bvn_pdf(gcc.bmc, x)
      f_pre * gcc.e_dev_i

      # Alternatively: test through accum_galaxy_pos!
      # elbo_vars_fd = ElboDeriv.ElboIntermediateVariables(T, 1, 1);
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
    v = ElboDeriv.get_bvn_derivs!(elbo_vars, gcc.bmc, x, true);
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
    ad_hess_gal = ForwardDiff.hessian(f_wrap_gal, par_gal)
    @test_approx_eq ad_hess_gal fs1m.h
  end
end


function test_fs0m_derivatives()
  # TODO: test with a real and asymmetric wcs jacobian.
  blob, mp, three_bodies = gen_three_body_dataset();
  omitted_ids = Int64[];
  kept_ids = setdiff(1:length(ids), omitted_ids);

  s = 1
  b = 1

  patch = mp.patches[s, b];
  u = mp.vp[s][ids.u]
  u_pix = WCS.world_to_pixel(
    patch.wcs_jacobian, patch.center, patch.pixel_center, u)
  x = ceil(u_pix + [1.0, 2.0])

  elbo_vars = ElboDeriv.ElboIntermediateVariables(Float64, 1, 1);

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
    elbo_vars_fd = ElboDeriv.ElboIntermediateVariables(T, 1, 1);
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


function test_bvn_derivatives()
  # Test log(bvn prob) / d(mean, sigma)

  x = Float64[2.0, 3.0]

  e_angle, e_axis, e_scale = (1.1, 0.02, 4.8)
  sigma = Util.get_bvn_cov(e_axis, e_angle, e_scale)

  offset = Float64[0.5, 0.25]

  # Note that get_bvn_derivs doesn't use the weight, so set it to something
  # strange to check that it doesn't matter.
  #weight = 0.724
  weight = 1.0

  bvn = ElboDeriv.BvnComponent(offset, sigma, weight);
  elbo_vars = ElboDeriv.ElboIntermediateVariables(Float64, 1, 1);
  v = ElboDeriv.get_bvn_derivs!(elbo_vars, bvn, x, true);

  function bvn_function{T <: Number}(x::Vector{T}, sigma::Matrix{T})
    local_x = offset - x
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
    bvn_function(x_loc, sig_loc)
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


function test_galaxy_variable_transform()
  # This is testing transform_bvn_derivs!

  # TODO: test with a real and asymmetric wcs jacobian.
  # We only need this for a psf and jacobian.
  blob, mp, three_bodies = gen_three_body_dataset();

  # Pick a single source and band for testing.
  s = 1
  b = 5

  # The pixel and world centers shouldn't matter for derivatives.
  patch = mp.patches[s, b];
  psf = patch.psf[1];

  # Pick out a single galaxy component for testing.
  gp = galaxy_prototypes[2][4];
  e_dev_dir = -1.0;
  e_dev_i = 0.85;

  # Test the variable transformation.
  e_angle, e_axis, e_scale = (1.1, 0.02, 4.8)

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


  function f_bvn_wrap{T <: Number}(par::Vector{T})
    u = par[par_ids_u]
    e_angle = par[par_ids_e_angle]
    e_axis = par[par_ids_e_axis]
    e_scale = par[par_ids_e_scale]
    u_pix = WCS.world_to_pixel(
      patch.wcs_jacobian, patch.center, patch.pixel_center, u)

    sigma = Util.get_bvn_cov(e_axis, e_angle, e_scale)

    function bvn_function{T <: Number}(u_pix::Vector{T}, sigma::Matrix{T})
      local_x = x - u_pix
      -0.5 * ((local_x' * (sigma \ local_x))[1,1] + log(det(sigma)))
    end

    bvn_function(u_pix, sigma)
  end

  # First just test the bvn function itself
  par = wrap_par(u, e_angle, e_axis, e_scale)
  u_pix = WCS.world_to_pixel(
    patch.wcs_jacobian, patch.center, patch.pixel_center, u)
  sigma = Util.get_bvn_cov(e_axis, e_angle, e_scale)
  bmc = ElboDeriv.BvnComponent(u_pix, sigma, 1.0);
  sig_sf = ElboDeriv.GalaxySigmaDerivs(e_angle, e_axis, e_scale, sigma);
  gcc = ElboDeriv.GalaxyCacheComponent(1.0, 1.0, bmc, sig_sf);
  elbo_vars = ElboDeriv.ElboIntermediateVariables(Float64, 1, 1);
  ElboDeriv.get_bvn_derivs!(elbo_vars, bmc, x, true);
  ElboDeriv.transform_bvn_derivs!(elbo_vars, gcc, patch.wcs_jacobian);

  # When the hessian is large the result has the wrong sign.  Maybe that
  # sign is wrong?
  elbo_vars.bvn_sigsig_h
  elbo_vars.bvn_sig_d

  f_bvn_wrap(par)

  # Check the gradient.
  ad_grad = ForwardDiff.gradient(f_bvn_wrap, par);
  @test_approx_eq ad_grad [elbo_vars.bvn_u_d; elbo_vars.bvn_s_d]

  ad_hess = ForwardDiff.hessian(f_bvn_wrap, par);
  @test_approx_eq ad_hess[1:2, 1:2] elbo_vars.bvn_uu_h
  @test_approx_eq ad_hess[1:2, 3:5] elbo_vars.bvn_us_h

  celeste_bvn_ss_h = deepcopy(elbo_vars.bvn_ss_h);
  ad_bvn_ss_h = deepcopy(ad_hess[3:5, 3:5])
  @test_approx_eq ad_hess[3:5, 3:5] elbo_vars.bvn_ss_h
end


function test_galaxy_cache_component()
  # TODO: eliminate some of the redundancy in these tests.

  # TODO: test with a real and asymmetric wcs jacobian.
  # We only need this for a psf and jacobian.
  blob, mp, three_bodies = gen_three_body_dataset();

  # Pick a single source and band for testing.
  s = 1
  b = 5

  # The pixel and world centers shouldn't matter for derivatives.
  patch = mp.patches[s, b];
  psf = patch.psf[1];

  # Pick out a single galaxy component for testing.
  gp = galaxy_prototypes[2][4];
  e_dev_dir = -1.0;
  e_dev_i = 0.85;

  # Test the variable transformation.
  e_angle, e_axis, e_scale = (1.1, 0.02, 4.8)

  u = Float64[5.3, 2.9]
  x = Float64[7.0, 5.0]

  # The indices in par of each variable.
  par_ids_u = [1, 2]
  par_ids_e_axis = 3
  par_ids_e_angle = 4
  par_ids_e_scale = 5
  par_ids_length = 5

  function f_wrap{T <: Number}(par::Vector{T})
    u = par[par_ids_u]
    e_angle = par[par_ids_e_angle]
    e_axis = par[par_ids_e_axis]
    e_scale = par[par_ids_e_scale]
    u_pix = WCS.world_to_pixel(
      patch.wcs_jacobian, patch.center, patch.pixel_center, u)
    elbo_vars_fd = ElboDeriv.ElboIntermediateVariables(T, 1, 1)
    e_dev_i_fd = convert(T, e_dev_i)
    gcc = ElboDeriv.GalaxyCacheComponent(
            e_dev_dir, e_dev_i_fd, gp, psf,
            u_pix, e_axis, e_angle, e_scale, false);

    py1, py2, f_pre = ElboDeriv.eval_bvn_pdf(gcc.bmc, x);

    log(f_pre)
  end

  function wrap_par{T <: Number}(
      u::Vector{T}, e_angle::T, e_axis::T, e_scale::T)
    par = zeros(T, par_ids_length)
    par[par_ids_u] = u
    par[par_ids_e_angle] = e_angle
    par[par_ids_e_axis] = e_axis
    par[par_ids_e_scale] = e_scale
    par
  end

  par = wrap_par(u, e_angle, e_axis, e_scale)
  u_pix = WCS.world_to_pixel(
    patch.wcs_jacobian, patch.center, patch.pixel_center, u)
  gcc = ElboDeriv.GalaxyCacheComponent(
          e_dev_dir, e_dev_i, gp, psf, u_pix, e_axis, e_angle, e_scale, true);
  elbo_vars = ElboDeriv.ElboIntermediateVariables(Float64, 1, 1);
  ElboDeriv.get_bvn_derivs!(elbo_vars, gcc.bmc, x, true);
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
  celeste_bvn_ss_h = deepcopy(elbo_vars.bvn_ss_h);
  ad_bvn_ss_h = deepcopy(ad_hess[3:5, 3:5])
  @test_approx_eq ad_hess[3:5, 3:5] elbo_vars.bvn_ss_h

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


function test_brightness_hessian()
  blob, mp, star_cat = gen_sample_star_dataset();
  kept_ids = [ ids.r1; ids.r2; ids.c1[:]; ids.c2[:] ];
  omitted_ids = setdiff(1:length(ids), kept_ids);
  i = 1

  for squares in [false, true], b in 1:5, i in 1:2
    squares_string = squares ? "E_G" : "E_G2"
    println("Testing brightness $(squares_string) for band $b, type $i")
    function wrap_source_brightness{NumType <: Number}(
        vp::Vector{NumType}, calculate_derivs::Bool)
      ret = zero_sensitive_float(CanonicalParams, NumType);
      sb = ElboDeriv.SourceBrightness(vp, calculate_derivs=calculate_derivs);
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
      wrap_source_brightness(vp, false).v
    end

    bright = wrap_source_brightness(mp.vp[1], true);

    @test_approx_eq bright.v wrap_source_brightness_value(mp.vp[1]);

    ad_grad = ForwardDiff.gradient(wrap_source_brightness_value, mp.vp[1]);
    @test_approx_eq ad_grad bright.d[:, 1]

    ad_hess_fun = ForwardDiff.hessian(wrap_source_brightness_value);
    ad_hess = ad_hess_fun(mp.vp[1]);
    @test_approx_eq ad_hess bright.h
  end
end


function test_dsiginv_dsig()
  e_angle, e_axis, e_scale = (1.1, 0.02, 4.8) # elbo_vars.bvn_sigsig_h is large
  the_cov = Util.get_bvn_cov(e_axis, e_angle, e_scale)
  the_mean = Float64[0., 0.]
  bvn = ElboDeriv.BvnComponent(the_mean, the_cov, 1.0);
  sigma_vec = Float64[ the_cov[1, 1], the_cov[1, 2], the_cov[2, 2] ]

  for component_index = 1:3
    components = [(1, 1), (1, 2), (2, 2)]
    function invert_sigma{NumType <: Number}(sigma_vec::Vector{NumType})
      sigma_loc = NumType[sigma_vec[1] sigma_vec[2]; sigma_vec[2] sigma_vec[3]]
      sigma_inv = inv(sigma_loc)
      sigma_inv[components[component_index]...]
    end

    ad_grad = ForwardDiff.gradient(invert_sigma, sigma_vec);
    @test_approx_eq ad_grad bvn.dsiginv_dsig[component_index, :][:]
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


test_dsiginv_dsig()
test_set_hess()
test_brightness_hessian()
test_bvn_derivatives()
test_galaxy_sigma_derivs()
test_galaxy_variable_transform()
test_galaxy_cache_component()
test_bvn_derivatives()
test_fs0m_derivatives()
test_fs1m_derivatives()
test_e_g_s_functions()
test_combine_pixel_sources()
test_add_log_term()
