using Celeste: ElboDeriv, SensitiveFloats

using Base.Test
using Distributions


function true_star_init()
    blob, ea, body = gen_sample_star_dataset(perturb=false)

    ea.vp[1][ids.a] = [ 1.0 - 1e-4, 1e-4 ]
    ea.vp[1][ids.r2] = 1e-4
    ea.vp[1][ids.r1] = log(sample_star_fluxes[3]) - 0.5 * ea.vp[1][ids.r2]
    #ea.vp[1][ids.r1] = sample_star_fluxes[3] ./ ea.vp[1][ids.r2]
    ea.vp[1][ids.c2] = 1e-4

    blob, ea, body
end


"""
Return a vector of (h, w) indices of tiles that contain this source.
"""
function find_source_tiles(s::Int, b::Int, ea::ElboArgs)
    [ind2sub(size(ea.tile_source_map[b]), ind) for ind in
        find([ s in sources for sources in ea.tile_source_map[b]])]
end

#################################

function test_kl_divergence_values()
    blob, ea, three_bodies = gen_three_body_dataset();

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

    vs = ea.vp[s]

    # a
    q_a = Bernoulli(vs[ids.a[2]])
    p_a = Bernoulli(prior.a[2])
    test_kl(q_a, p_a, () -> ElboDeriv.subtract_kl_a(ea.vp[s]))

    # k
    q_k = Categorical(vs[ids.k[:, i]])
    p_k = Categorical(prior.k[:, i])
    function sklk()
        @assert i == 1
        ElboDeriv.subtract_kl_k(i, ea.vp[s]) / vs[ids.a[i]]
    end
    test_kl(q_k, p_k, sklk)

    # c
    old_c_mean = deepcopy(prior.c_mean)
    old_c_cov = deepcopy(prior.c_cov)
    prior.c_mean[:,d,i] = vs[ids.c1[:, i]]
    prior.c_cov[:,:,d,i] = diagm(vs[ids.c2[:, i]])
    q_c = MvNormal(vs[ids.c1[:, i]], diagm(vs[ids.c2[:, i]]))
    p_c = MvNormal(prior.c_mean[:, d, i], prior.c_cov[:, :, d, i])
    function sklc()
        ElboDeriv.subtract_kl_c(d, i, ea.vp[s]) /
          vs[ids.a[i]] * vs[ids.k[d, i]]
    end
    test_kl(q_c, p_c, sklc)
    prior.c_mean[:,:,:] = old_c_mean
    prior.c_cov[:,:,:] = old_c_cov

    # r
    q_r = Normal(vs[ids.r1[i]], sqrt(vs[ids.r2[i]]))
    p_r = Normal(prior.r_mean[i], sqrt(prior.r_var[i]))
    function sklr()
        @assert i == 1
        ElboDeriv.subtract_kl_r(i, ea.vp[s]) / vs[ids.a[i]]
    end
    test_kl(q_r, p_r, sklr)

end


function test_that_variance_is_low()
    # very peaked variational distribution---variance for F(m) should be low
    blob, ea, body = true_star_init();

    test_b = 3
    tile = ea.images[test_b].tiles[1,1];
    tile_source_map = ea.tile_source_map[test_b][1,1];

    h, w = 10, 12
    star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(ea, tile.b);
    sbs = ElboDeriv.SourceBrightness{Float64}[
      ElboDeriv.SourceBrightness(ea.vp[s]) for s in 1:ea.S];

    elbo_vars = ElboDeriv.ElboIntermediateVariables(
      Float64, ea.S, length(ea.active_sources));

    clear!(elbo_vars.E_G);
    clear!(elbo_vars.var_G);
    ElboDeriv.get_expected_pixel_brightness!(
      elbo_vars, h, w, sbs, star_mcs, gal_mcs, tile, ea, tile_source_map);

    @test 0 < elbo_vars.var_G.v[1] < 1e-2 * elbo_vars.E_G.v[1]^2
end


function test_that_star_truth_is_most_likely()
    blob, ea, body = true_star_init();
    best = ElboDeriv.elbo_likelihood(ea);

    for bad_a in [.3, .5, .9]
        ea_a = deepcopy(ea)
        ea_a.vp[1][ids.a] = [ 1.0 - bad_a, bad_a ]
        bad_a_lik = ElboDeriv.elbo_likelihood(ea_a)
        @test best.v[1] > bad_a_lik.v[1]
    end

    for h2 in -2:2
        for w2 in -2:2
            if !(h2 == 0 && w2 == 0)
                ea_mu = deepcopy(ea)
                ea_mu.vp[1][ids.u] += [h2 * .5, w2 * .5]
                bad_mu = ElboDeriv.elbo_likelihood(ea_mu)
                @test best.v[1] > bad_mu.v[1]
            end
        end
    end

    for delta in [.7, .9, 1.1, 1.3]
        ea_r1 = deepcopy(ea)
        ea_r1.vp[1][ids.r1] += log(delta)
        bad_r1 = ElboDeriv.elbo_likelihood(ea_r1)
        @test best.v[1] > bad_r1.v[1]
    end

    for b in 1:4
        for delta in [-.3, .3]
            ea_c1 = deepcopy(ea)
            ea_c1.vp[1][ids.c1[b, 1]] += delta
            bad_c1 = ElboDeriv.elbo_likelihood(ea_c1)
            @test best.v[1] > bad_c1.v[1]
        end
    end

end


function test_that_galaxy_truth_is_most_likely()
    blob, ea, body = gen_sample_galaxy_dataset(perturb=false);
    ea.vp[1][ids.a] = [ 0.01, .99 ]
    best = ElboDeriv.elbo_likelihood(ea);

    for bad_a in [.3, .5, .9]
        ea_a = deepcopy(ea);
        ea_a.vp[1][ids.a] = [ 1.0 - bad_a, bad_a ];
        bad_a =
          ElboDeriv.elbo_likelihood(ea_a; calculate_derivs=false);
        @test best.v[1] > bad_a.v[1];
    end

    for h2 in -2:2
        for w2 in -2:2
            if !(h2 == 0 && w2 == 0)
                ea_mu = deepcopy(ea)
                ea_mu.vp[1][ids.u] += [h2 * .5, w2 * .5]
                bad_mu = ElboDeriv.elbo_likelihood(
                  ea_mu; calculate_derivs=false)
                @test best.v[1] > bad_mu.v[1]
            end
        end
    end

    for bad_scale in [.8, 1.2]
        ea_r1 = deepcopy(ea)
        ea_r1.vp[1][ids.r1] += 2 * log(bad_scale)
        bad_r1 = ElboDeriv.elbo_likelihood(
          ea_r1; calculate_derivs=false)
        @test best.v[1] > bad_r1.v[1]
    end

    for n in [:e_axis, :e_angle, :e_scale]
        for bad_scale in [.8, 1.2]
            ea_bad = deepcopy(ea)
            ea_bad.vp[1][ids.(n)] *= bad_scale
            bad_elbo = ElboDeriv.elbo_likelihood(
              ea_bad; calculate_derivs=false)
            @test best.v[1] > bad_elbo.v[1]
        end
    end

    # TODO: this test is failling because the stamps use three psf
    # components.
    for b in 1:4
        println("b: $b")
        for delta in [-.3, .3]
            println("delta: $delta")
            ea_c1 = deepcopy(ea)
            ea_c1.vp[1][ids.c1[b, 2]] += delta
            bad_c1 = ElboDeriv.elbo_likelihood(
              ea_c1; calculate_derivs=false)
            @test best.v[1] > bad_c1.v[1]
        end
    end
end


function test_coadd_cat_init_is_most_likely()  # on a real stamp
    stamp_id = "5.0073-0.0739_2kpsf"
    blob = SampleData.load_stamp_blob(datadir, stamp_id);

    cat_entries = SampleData.load_stamp_catalog(datadir, "s82-$stamp_id", blob);
    bright(ce) = sum(ce.star_fluxes) > 3 || sum(ce.gal_fluxes) > 3
    cat_entries = filter(bright, cat_entries)

    ce_pix_locs =
      [ [ WCSUtils.world_to_pix(blob[b].wcs, ce.pos) for b=1:5 ]
        for ce in cat_entries ]

    function ce_inbounds(ce)
        pix_locs = [ WCSUtils.world_to_pix(blob[b].wcs, ce.pos) for b=1:5 ]
        inbounds(pos) = pos[1] > -10. && pos[2] > -10 &&
                        pos[1] < 61 && pos[2] < 61
        reduce(&, [inbounds(pos) for pos in pix_locs])
    end
    cat_entries = filter(ce_inbounds, cat_entries)

    ea = make_elbo_args(blob, cat_entries)
    for s in 1:length(cat_entries)
        ea.vp[s][ids.a[2]] = cat_entries[s].is_star ? 0.01 : 0.99
        ea.vp[s][ids.a[1]] = 1.0 - ea.vp[s][ids.a[2]]
    end
    best = ElboDeriv.elbo_likelihood(ea; calculate_derivs=false);

    # s is the brightest source.
    s = 1

    for bad_scale in [.7, 1.3]
        ea_r1 = deepcopy(ea)
        ea_r1.vp[s][ids.r1] += 2 * log(bad_scale)
        bad_r1 = ElboDeriv.elbo_likelihood(
          ea_r1; calculate_derivs=false)
        @test best.v[1] > bad_r1.v[1]
    end

    for n in [:e_axis, :e_angle, :e_scale]
        for bad_scale in [.6, 1.8]
            ea_bad = deepcopy(ea)
            ea_bad.vp[s][ids.(n)] *= bad_scale
            bad_elbo = ElboDeriv.elbo_likelihood(
              ea_bad; calculate_derivs=false)
            @test best.v[1] > bad_elbo.v[1]
        end
    end

    for bad_a in [.3, .7]
        ea_a = deepcopy(ea)
        ea_a.vp[s][ids.a] = [ 1.0 - bad_a, bad_a ]

        bad_a = ElboDeriv.elbo_likelihood(
          ea_a; calculate_derivs=false)
        @test best.v[1] > bad_a.v[1]
    end

    for h2 in -2:2
        for w2 in -2:2
            if !(h2 == 0 && w2 == 0)
                ea_mu = deepcopy(ea)
                ea_mu.vp[s][ids.u] += [0.5h2, 0.5w2]
                bad_mu = ElboDeriv.elbo_likelihood(
                  ea_mu; calculate_derivs=false)
                @test best.v[1] > bad_mu.v[1]
            end
        end
    end

    for b in 1:4
        for delta in [-2., 2.]
            ea_c1 = deepcopy(ea)
            ea_c1.vp[s][ids.c1[b, :]] += delta
            bad_c1 = ElboDeriv.elbo_likelihood(
              ea_c1; calculate_derivs=false)
            info("$(best.v[1])  >  $(bad_c1.v[1])")
            @test best.v[1] > bad_c1.v[1]
        end
    end
end


function test_tiny_image_tiling()
  # Test that the tilling doesn't matter much for a body that is nearly a
  # point with a narrow psf.

  blob0 = SampleData.load_stamp_blob(datadir, "164.4311-39.0359_2kpsf");
  pc = PsfComponent(1./3, zeros(2), 1e-4 * eye(2));
  trivial_psf = [pc, pc, pc]
  pixels = ones(100, 1) * 12
  pixels[98:100, 1] = [1e3, 1e4, 1e5]
  img = Image(3, 1, pixels, 1, blob0[3].wcs, trivial_psf, 1, 1, 1,
              fill(3., size(pixels)), fill(4., size(pixels, 1)),
              blob0[3].raw_psf_comp);
  catalog = [sample_ce([100., 1], true),];
  catalog[1].star_fluxes = ones(5) * 1e5

  ea0 = make_elbo_args(
    [img], catalog, patch_radius=Inf)

  println(size(ea0.patches))
  elbo_lik = ElboDeriv.elbo_likelihood(ea0;
        calculate_derivs=false, calculate_hessian=false);

  tile_width = 2
  ea1 = make_elbo_args(
    [img], catalog, tile_width=tile_width, patch_radius=10.);
  elbo_lik_tiles =
    ElboDeriv.elbo_likelihood(
      ea1, calculate_derivs=false, calculate_hessian=false);

  tile_width = 5
  ea2 = make_elbo_args(
      [img], catalog, tile_width=tile_width, patch_radius=10.);
  elbo_lik_tiles2 =
    ElboDeriv.elbo_likelihood(
      ea2, calculate_derivs=false, calculate_hessian=false);

  @test_approx_eq elbo_lik_tiles.v[1] elbo_lik_tiles2.v[1]
  @test_approx_eq_eps elbo_lik.v[1] elbo_lik_tiles.v[1] 100.

end


function test_elbo_with_nan()
    blob, ea, body = gen_sample_star_dataset(perturb=false);

    for b in 1:5
        blob[b].pixels[1,1] = NaN
    end

    # Set tile width to 5 to test the code for tiles with no sources.
    ea = make_elbo_args(blob, body, tile_width=5);
    initial_elbo = ElboDeriv.elbo(ea; calculate_hessian=false);

    nan_elbo = ElboDeriv.elbo(ea);

    # We deleted a pixel, so there's reason to expect them to be different,
    # but importantly they're reasonably close and not NaN.
    @test_approx_eq_eps (nan_elbo.v[1] - initial_elbo.v[1]) / initial_elbo.v[1] 0. 1e-4
    deriv_rel_err = (nan_elbo.d - initial_elbo.d) ./ initial_elbo.d
    @test_approx_eq_eps deriv_rel_err fill(0., length(ea.vp[1])) 0.05
end


function test_trim_source_tiles()
  # Set a seed to avoid a flaky test.
  blob, ea, bodies = gen_n_body_dataset(3, seed=42);

  # With the above seed, this is near the middle of the image.
  ea.active_sources = [1]
  trimmed_tiled_blob = 
      Infer.trim_source_tiles(ea; noise_fraction=0.1);
  loc_ids = ids.u
  non_loc_ids = setdiff(1:length(ids), ids.u)
  for b=1:length(blob)
    println("Testing b = $b")
    # Make sure pixels got NaN-ed out
    @test(
      sum([ sum(!Base.isnan(tile.pixels)) for tile in trimmed_tiled_blob[b].tiles]) <
      sum([ sum(!Base.isnan(tile.pixels)) for tile in ea.images[b].tiles]))
    s_tiles = find_source_tiles(s, b, ea)
    ea.active_sources = [s];
    elbo_full = ElboDeriv.elbo(ea; calculate_hessian=false);

    ea_trimmed = deepcopy(ea)
    ea_trimmed.images = trimmmed_tiled_blob
    elbo_trim = ElboDeriv.elbo(ea2; calculate_hessian=false);
    @test_approx_eq_eps(
      elbo_full.d[loc_ids, 1] ./ elbo_trim.d[loc_ids, 1],
      fill(1.0, length(loc_ids)), 0.06)
    @test_approx_eq_eps(
      elbo_full.d[non_loc_ids, 1] ./ elbo_trim.d[non_loc_ids, 1],
      fill(1.0, length(non_loc_ids)), 4e-3)
  end

  # Test min_radius_pix on just one tile.
  b = 3
  s_tiles = find_source_tiles(s, b, ea)

  # Set the source to be very dim:
  ea.vp[s][ids.r1] = 0.01
  ea.vp[s][ids.r2] = 0.01

  min_radius_pix = 6.0
  trimmed_tiled_blob = 
      Infer.trim_source_tiles(
        ea, noise_fraction=0.1, min_radius_pix = min_radius_pix);

  total_nonempty_pixels = 0.0
  for tile_index in s_tiles
    tile = trimmed_tiled_blob[b].tiles[tile_index...]
    total_nonempty_pixels += sum(!Base.isnan(tile.pixels))
  end
  @test_approx_eq_eps total_nonempty_pixels pi * min_radius_pix ^ 2 2.0

  min_radius_pix = 0.0
  trimmed_tiled_blob = 
      Infer.trim_source_tiles(
        s, ea, trimmed_tiled_blob, noise_fraction=0.1,
            min_radius_pix = min_radius_pix);

  total_nonempty_pixels = 0.0
  for tile_index in s_tiles
    tile = trimmed_tiled_blob[b].tiles[tile_index...]
    total_nonempty_pixels += sum(!Base.isnan(tile.pixels))
  end
  @test total_nonempty_pixels == 0.0

end


####################################################

#test_trim_source_tiles()
test_tiny_image_tiling()
test_kl_divergence_values()
test_that_variance_is_low()
test_that_star_truth_is_most_likely()
test_that_galaxy_truth_is_most_likely()
test_coadd_cat_init_is_most_likely()
test_elbo_with_nan()
