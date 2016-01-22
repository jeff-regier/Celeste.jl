using Celeste
using CelesteTypes
using Base.Test
using Distributions
using SampleData

import SloanDigitalSkySurvey: SDSS
import SloanDigitalSkySurvey: WCS

println("Running ELBO value tests.")


function true_star_init()
    blob, mp, body, tiled_blob = gen_sample_star_dataset(perturb=false)

    mp.vp[1][ids.a] = [ 1.0 - 1e-4, 1e-4 ]
    mp.vp[1][ids.r2] = 1e-4
    mp.vp[1][ids.r1] = log(sample_star_fluxes[3]) - 0.5 * mp.vp[1][ids.r2]
    #mp.vp[1][ids.r1] = sample_star_fluxes[3] ./ mp.vp[1][ids.r2]
    mp.vp[1][ids.c2] = 1e-4

    blob, mp, body, tiled_blob
end


#################################

function test_kl_divergence_values()
    blob, mp, three_bodies, tiled_blob = gen_three_body_dataset();

    s = 1
    i = 1
    d = 1
    sample_size = 2_000_000

    function test_kl(q_dist, p_dist, kl_fun)
        q_samples = rand(q_dist, sample_size)
        empirical_kl_samples =
          logpdf(q_dist, q_samples) - logpdf(p_dist, q_samples)
        empirical_kl = mean(empirical_kl_samples)
        exact_kl = -kl_fun()
        tol = 4 * std(empirical_kl_samples) / sqrt(sample_size)
        min_diff = 1e-2 * std(empirical_kl_samples) / sqrt(sample_size)

        # TODO: fix this test, which assumes an in-place update.
        @test_approx_eq_eps empirical_kl exact_kl tol
    end

    vs = mp.vp[s]

    # a
    q_a = Bernoulli(vs[ids.a[2]])
    p_a = Bernoulli(mp.pp.a[2])
    test_kl(q_a, p_a, () -> ElboDeriv.subtract_kl_a(mp.vp[s], mp.pp))

    # k
    q_k = Categorical(vs[ids.k[:, i]])
    p_k = Categorical(mp.pp.k[:, i])
    function sklk()
        @assert i == 1
        ElboDeriv.subtract_kl_k(i, mp.vp[s], mp.pp) / vs[ids.a[i]]
    end
    test_kl(q_k, p_k, sklk)

    # c
    mp.pp.c_mean[:,d,i] = vs[ids.c1[:, i]]
    mp.pp.c_cov[:,:,d,i] = diagm(vs[ids.c2[:, i]])
    q_c = MvNormal(vs[ids.c1[:, i]], diagm(vs[ids.c2[:, i]]))
    p_c = MvNormal(mp.pp.c_mean[:, d, i], mp.pp.c_cov[:, :, d, i])
    function sklc()
        ElboDeriv.subtract_kl_c(d, i, mp.vp[s], mp.pp) /
          vs[ids.a[i]] * vs[ids.k[d, i]]
    end
    test_kl(q_c, p_c, sklc)

    # r
    q_r = Normal(vs[ids.r1[i]], sqrt(vs[ids.r2[i]]))
    p_r = Normal(mp.pp.r_mean[i], sqrt(mp.pp.r_var[i]))
    function sklr()
        @assert i == 1
        ElboDeriv.subtract_kl_r(i, mp.vp[s], mp.pp) / vs[ids.a[i]]
    end
    test_kl(q_r, p_r, sklr)

end


function test_that_variance_is_low()
    # very peaked variational distribution---variance for F(m) should be low
    blob, mp, body, tiled_blob = true_star_init();

    test_b = 3
    tile = tiled_blob[test_b][1,1];
    tile_sources = mp.tile_sources[test_b][1,1];

    h, w = 10, 12
    star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(mp, tile.b);
    sbs = ElboDeriv.SourceBrightness{Float64}[
      ElboDeriv.SourceBrightness(mp.vp[s]) for s in 1:mp.S];

    elbo_vars = ElboDeriv.ElboIntermediateVariables(
      Float64, mp.S, length(mp.active_sources));

    clear!(elbo_vars.E_G);
    clear!(elbo_vars.var_G);
    ElboDeriv.get_expected_pixel_brightness!(
      elbo_vars, h, w, sbs, star_mcs, gal_mcs, tile, mp, tile_sources);

    @test 0 < elbo_vars.var_G.v < 1e-2 * elbo_vars.E_G.v^2
end


function test_that_star_truth_is_most_likely()
    blob, mp, body, tiled_blob = true_star_init();
    best = ElboDeriv.elbo_likelihood(tiled_blob, mp);

    for bad_a in [.3, .5, .9]
        mp_a = deepcopy(mp)
        mp_a.vp[1][ids.a] = [ 1.0 - bad_a, bad_a ]
        bad_a_lik = ElboDeriv.elbo_likelihood(tiled_blob, mp_a)
        @test best.v > bad_a_lik.v
    end

    for h2 in -2:2
        for w2 in -2:2
            if !(h2 == 0 && w2 == 0)
                mp_mu = deepcopy(mp)
                mp_mu.vp[1][ids.u] += [h2 * .5, w2 * .5]
                bad_mu = ElboDeriv.elbo_likelihood(tiled_blob, mp_mu)
                @test best.v > bad_mu.v
            end
        end
    end

    for delta in [.7, .9, 1.1, 1.3]
        mp_r1 = deepcopy(mp)
        mp_r1.vp[1][ids.r1] += log(delta)
        bad_r1 = ElboDeriv.elbo_likelihood(tiled_blob, mp_r1)
        @test best.v > bad_r1.v
    end

    for b in 1:4
        for delta in [-.3, .3]
            mp_c1 = deepcopy(mp)
            mp_c1.vp[1][ids.c1[b, 1]] += delta
            bad_c1 = ElboDeriv.elbo_likelihood(tiled_blob, mp_c1)
            @test best.v > bad_c1.v
        end
    end

end


function test_that_galaxy_truth_is_most_likely()
    blob, mp, body, tiled_blob = gen_sample_galaxy_dataset(perturb=false);
    mp.vp[1][ids.a] = [ 0.01, .99 ]
    best = ElboDeriv.elbo_likelihood(tiled_blob, mp);

    for bad_a in [.3, .5, .9]
        mp_a = deepcopy(mp);
        mp_a.vp[1][ids.a] = [ 1.0 - bad_a, bad_a ];
        bad_a =
          ElboDeriv.elbo_likelihood(tiled_blob, mp_a; calculate_derivs=false);
        @test best.v > bad_a.v;
    end

    for h2 in -2:2
        for w2 in -2:2
            if !(h2 == 0 && w2 == 0)
                mp_mu = deepcopy(mp)
                mp_mu.vp[1][ids.u] += [h2 * .5, w2 * .5]
                bad_mu = ElboDeriv.elbo_likelihood(
                  tiled_blob, mp_mu; calculate_derivs=false)
                @test best.v > bad_mu.v
            end
        end
    end

    for bad_scale in [.8, 1.2]
        mp_r1 = deepcopy(mp)
        mp_r1.vp[1][ids.r1] += 2 * log(bad_scale)
        bad_r1 = ElboDeriv.elbo_likelihood(
          tiled_blob, mp_r1; calculate_derivs=false)
        @test best.v > bad_r1.v
    end

    for n in [:e_axis, :e_angle, :e_scale]
        for bad_scale in [.8, 1.2]
            mp_bad = deepcopy(mp)
            mp_bad.vp[1][ids.(n)] *= bad_scale
            bad_elbo = ElboDeriv.elbo_likelihood(
              tiled_blob, mp_bad; calculate_derivs=false)
            @test best.v > bad_elbo.v
        end
    end

    for b in 1:4
        for delta in [-.3, .3]
            mp_c1 = deepcopy(mp)
            mp_c1.vp[1][ids.c1[b, 2]] += delta
            bad_c1 = ElboDeriv.elbo_likelihood(
              tiled_blob, mp_c1; calculate_derivs=false)
            @test best.v > bad_c1.v
        end
    end
end


function test_coadd_cat_init_is_most_likely()  # on a real stamp
    stamp_id = "5.0073-0.0739"
    blob = SkyImages.load_stamp_blob(dat_dir, stamp_id);

    cat_entries = SkyImages.load_stamp_catalog(dat_dir, "s82-$stamp_id", blob);
    bright(ce) = sum(ce.star_fluxes) > 3 || sum(ce.gal_fluxes) > 3
    cat_entries = filter(bright, cat_entries)

    ce_pix_locs =
      [ [ WCS.world_to_pixel(blob[b].wcs, ce.pos) for b=1:5 ]
        for ce in cat_entries ]

    function ce_inbounds(ce)
        pix_locs = [ WCS.world_to_pixel(blob[b].wcs, ce.pos) for b=1:5 ]
        inbounds(pos) = pos[1] > -10. && pos[2] > -10 &&
                        pos[1] < 61 && pos[2] < 61
        reduce(&, [inbounds(pos) for pos in pix_locs])
    end
    cat_entries = filter(ce_inbounds, cat_entries)

    tiled_blob, mp = ModelInit.initialize_celeste(blob, cat_entries)
    for s in 1:length(cat_entries)
        mp.vp[s][ids.a[2]] = cat_entries[s].is_star ? 0.01 : 0.99
        mp.vp[s][ids.a[1]] = 1.0 - mp.vp[s][ids.a[2]]
    end
    best = ElboDeriv.elbo_likelihood(tiled_blob, mp; calculate_derivs=false);

    # s is the brightest source.
    s = 1

    for bad_scale in [.7, 1.3]
        mp_r1 = deepcopy(mp)
        mp_r1.vp[s][ids.r1] += 2 * log(bad_scale)
        bad_r1 = ElboDeriv.elbo_likelihood(
          tiled_blob, mp_r1; calculate_derivs=false)
        @test best.v > bad_r1.v
    end

    for n in [:e_axis, :e_angle, :e_scale]
        for bad_scale in [.6, 1.8]
            mp_bad = deepcopy(mp)
            mp_bad.vp[s][ids.(n)] *= bad_scale
            bad_elbo = ElboDeriv.elbo_likelihood(
              tiled_blob, mp_bad; calculate_derivs=false)
            @test best.v > bad_elbo.v
        end
    end

    for bad_a in [.3, .7]
        mp_a = deepcopy(mp)
        mp_a.vp[s][ids.a] = [ 1.0 - bad_a, bad_a ]

        bad_a = ElboDeriv.elbo_likelihood(
          tiled_blob, mp_a; calculate_derivs=false)
        @test best.v > bad_a.v
    end

    for h2 in -2:2
        for w2 in -2:2
            if !(h2 == 0 && w2 == 0)
                mp_mu = deepcopy(mp)
                mp_mu.vp[s][ids.u] += [0.5h2, 0.5w2]
                bad_mu = ElboDeriv.elbo_likelihood(
                  tiled_blob, mp_mu; calculate_derivs=false)
                @test best.v > bad_mu.v
            end
        end
    end

    for b in 1:4
        for delta in [-2., 2.]
            mp_c1 = deepcopy(mp)
            mp_c1.vp[s][ids.c1[b, :]] += delta
            bad_c1 = ElboDeriv.elbo_likelihood(
              tiled_blob, mp_c1; calculate_derivs=false)
            info("$(best.v)  >  $(bad_c1.v)")
            @test best.v > bad_c1.v
        end
    end
end


function test_tiny_image_tiling()
  # Test that the tilling doesn't matter much for a body that is nearly a
  # point with a narrow psf.

  blob0 = SkyImages.load_stamp_blob(dat_dir, "164.4311-39.0359");
  pc = PsfComponent(1./3, zeros(2), 1e-4 * eye(2))
  trivial_psf = [pc, pc, pc]
  pixels = ones(100, 1) * 12
  pixels[98:100, 1] = [1e3, 1e4, 1e5]
  img = Image(3, 1, pixels, 3, blob0[3].wcs, 3., 4., trivial_psf, 1, 1, 1)
  catalog = [sample_ce([100., 1], true),]
  catalog[1].star_fluxes = ones(5) * 1e5


  tiled_blob0, mp0 = ModelInit.initialize_celeste(
    fill(img, 5), catalog, patch_radius=Inf)

  # These will be reused for all the subsequent tests because only
  # the tile sources change.
  elbo_vars = ElboDeriv.ElboIntermediateVariables(
    Float64, mp0.S, length(mp0.active_sources), calculate_derivs=false);
  sbs = ElboDeriv.load_source_brightnesses(mp0, calculate_derivs=false);

  ElboDeriv.elbo_likelihood!(elbo_vars, tiled_blob0[3], mp0, 3, sbs);
  elbo_lik = deepcopy(elbo_vars.elbo);

  tile_width = 2
  tiled_blob1, mp0 = ModelInit.initialize_celeste(
    fill(img, 5), catalog, tile_width=tile_width, patch_radius=10.)
  ElboDeriv.elbo_likelihood!(elbo_vars, tiled_blob0[3], mp0, 3, sbs);
  elbo_lik_tiles = deepcopy(elbo_vars.elbo);

  tile_width = 5
  tiled_blob2, mp0 =
    ModelInit.initialize_celeste(
      fill(img, 5), catalog, tile_width=tile_width, patch_radius=10.);
  ElboDeriv.elbo_likelihood!(elbo_vars, tiled_blob0[3], mp0, 3, sbs);
  elbo_lik_tiles2 = deepcopy(elbo_vars.elbo);

  @test_approx_eq elbo_lik_tiles.v elbo_lik_tiles2.v
  @test_approx_eq_eps elbo_lik.v elbo_lik_tiles.v 100.

end


function test_elbo_with_nan()
    blob, mp, body = gen_sample_star_dataset(perturb=false);

    # Set tile width to 5 to test the code for tiles with no sources.
    tiled_blob, mp = ModelInit.initialize_celeste(blob, body, tile_width=5);
    initial_elbo = ElboDeriv.elbo(tiled_blob, mp; calculate_hessian=false);

    for b in 1:5
        blob[b].pixels[1,1] = NaN
    end

    nan_elbo = ElboDeriv.elbo(tiled_blob, mp);

    # We deleted a pixel, so there's reason to expect them to be different,
    # but importantly they're reasonably close and not NaN.
    @test_approx_eq_eps (nan_elbo.v - initial_elbo.v) / initial_elbo.v 0. 1e-4
    deriv_rel_err = (nan_elbo.d - initial_elbo.d) ./ initial_elbo.d
    @test_approx_eq_eps deriv_rel_err fill(0., length(mp.vp[1])) 0.05
end


function test_trim_source_tiles()
  # Set a seed to avoid a flaky test.
  blob, mp, bodies, tiled_blob = gen_n_body_dataset(3, seed=42);

  # With the above seed, this is near the middle of the image.
  s = 1
  trimmed_tiled_blob = ModelInit.trim_source_tiles(
    s, mp, tiled_blob, noise_fraction=0.1);
  loc_ids = ids.u
  non_loc_ids = setdiff(1:length(ids), ids.u)
  for b=1:5
    println("Testing b = $b")
    # Make sure pixels got NaN-ed out
    @test(
      sum([ sum(!Base.isnan(tile.pixels)) for tile in trimmed_tiled_blob[b]]) <
      sum([ sum(!Base.isnan(tile.pixels)) for tile in tiled_blob[b]]))
    s_tiles = SkyImages.find_source_tiles(s, b, mp)
    mp.active_sources = [s];
    elbo_full = ElboDeriv.elbo(tiled_blob, mp; calculate_hessian=false);
    elbo_trim = ElboDeriv.elbo(trimmed_tiled_blob, mp; calculate_hessian=false);
    @test_approx_eq_eps(
      elbo_full.d[loc_ids, 1] ./ elbo_trim.d[loc_ids, 1],
      fill(1.0, length(loc_ids)), 0.06)
    @test_approx_eq_eps(
      elbo_full.d[non_loc_ids, 1] ./ elbo_trim.d[non_loc_ids, 1],
      fill(1.0, length(non_loc_ids)), 2e-3)
  end
end



####################################################

test_kl_divergence_values()
test_that_variance_is_low()
test_that_star_truth_is_most_likely()
test_that_galaxy_truth_is_most_likely()
test_coadd_cat_init_is_most_likely()
test_tiny_image_tiling()
test_elbo_with_nan()
test_trim_source_tiles()
