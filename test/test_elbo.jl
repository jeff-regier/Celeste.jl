using Celeste: DeterministicVI, SensitiveFloats

using Base.Test
using Distributions
using DerivativeTestUtils

######################################
# Helper functions

function true_star_init()
    blob, ea, body = gen_sample_star_dataset(perturb=false)

    ea.vp[1][ids.a[:, 1]] = [ 1.0 - 1e-4, 1e-4 ]
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

function test_set_hess()
    sf = zero_sensitive_float(CanonicalParams)
    set_hess!(sf, 2, 3, 5.0)
    @test_approx_eq sf.h[2, 3] 5.0
    @test_approx_eq sf.h[3, 2] 5.0

    set_hess!(sf, 4, 4, 6.0)
    @test_approx_eq sf.h[4, 4] 6.0
end


function test_bvn_cov()
        e_axis = .7
        e_angle = pi/5
        e_scale = 2.

        manual_11 = e_scale^2 * (1 + (e_axis^2 - 1) * (sin(e_angle))^2)
        util_11 = DeterministicVI.get_bvn_cov(e_axis, e_angle, e_scale)[1,1]
        @test_approx_eq util_11 manual_11

        manual_12 = e_scale^2 * (1 - e_axis^2) * (cos(e_angle)sin(e_angle))
        util_12 = DeterministicVI.get_bvn_cov(e_axis, e_angle, e_scale)[1,2]
        @test_approx_eq util_12 manual_12

        manual_22 = e_scale^2 * (1 + (e_axis^2 - 1) * (cos(e_angle))^2)
        util_22 = DeterministicVI.get_bvn_cov(e_axis, e_angle, e_scale)[2,2]
        @test_approx_eq util_22 manual_22
end


function test_tile_predicted_image()
    blob, ea, body = gen_sample_star_dataset(perturb=false)
    tile = ea.images[1].tiles[1, 1]
    tile_source_map = ea.tile_source_map[1][1, 1]
    pred_image =
        DeterministicVI.tile_predicted_image(tile, ea, tile_source_map; include_epsilon=true)

    # Regress the tile pixels onto the predicted image
    # TODO: Why isn't the regression closer to one?    Something in the sample data
    # generation?
    reg_coeff = dot(tile.pixels[:], pred_image[:]) / dot(pred_image[:], pred_image[:])
    residuals = pred_image * reg_coeff - tile.pixels
    residual_sd = sqrt(mean(residuals .^ 2))

    @test residual_sd / mean(tile.pixels) < 0.1
end


function test_derivative_flags()
    blob, ea, body = gen_two_body_dataset()

    elbo = DeterministicVI.elbo(ea)

    elbo_noderiv = DeterministicVI.elbo(ea; calculate_derivs=false)
    @test_approx_eq elbo.v[1] elbo_noderiv.v[1]
    @test_approx_eq elbo_noderiv.d zeros(size(elbo_noderiv.d))
    @test_approx_eq elbo_noderiv.h zeros(size(elbo_noderiv.h))

    elbo_nohess = DeterministicVI.elbo(ea; calculate_hessian=false)
    @test_approx_eq elbo.v[1] elbo_nohess.v
    @test_approx_eq elbo.d elbo_nohess.d
    @test_approx_eq elbo_noderiv.h zeros(size(elbo_noderiv.h))
end


function test_active_sources()
    # Test that the derivatives of the expected brightnesses partition in
    # active_sources.

    blob, ea, body = gen_two_body_dataset()
    b = 1
    tile = ea.images[b].tiles[1,1]
    h, w = 10, 10

    ea.active_sources = [1, 2]
    elbo_lik_12 = DeterministicVI.elbo_likelihood(ea)

    ea.active_sources = [1]
    elbo_lik_1 = DeterministicVI.elbo_likelihood(ea)

    ea.active_sources = [2]
    elbo_lik_2 = DeterministicVI.elbo_likelihood(ea)

    @test_approx_eq elbo_lik_12.v[1] elbo_lik_1.v
    @test_approx_eq elbo_lik_12.v[1] elbo_lik_2.v

    @test_approx_eq elbo_lik_12.d[:, 1] elbo_lik_1.d[:, 1]
    @test_approx_eq elbo_lik_12.d[:, 2] elbo_lik_2.d[:, 1]

    P = length(CanonicalParams)
    @test_approx_eq elbo_lik_12.h[1:P, 1:P] elbo_lik_1.h
    @test_approx_eq elbo_lik_12.h[(1:P) + P, (1:P) + P] elbo_lik_2.h
end


function test_that_variance_is_low()
    # very peaked variational distribution---variance for F(m) should be low
    blob, ea, body = true_star_init();

    test_b = 3
    tile = ea.images[test_b].tiles[1,1];
    tile_source_map = ea.tile_source_map[test_b][1,1];

    h, w = 10, 12
    star_mcs, gal_mcs = DeterministicVI.load_bvn_mixtures(ea, tile.b);
    sbs = DeterministicVI.SourceBrightness{Float64}[
      DeterministicVI.SourceBrightness(ea.vp[s]) for s in 1:ea.S];

    elbo_vars = DeterministicVI.ElboIntermediateVariables(
      Float64, ea.S, length(ea.active_sources));

    clear!(elbo_vars.E_G);
    clear!(elbo_vars.var_G);
    DeterministicVI.get_expected_pixel_brightness!(
      elbo_vars, h, w, sbs, star_mcs, gal_mcs, tile, ea, tile_source_map);

    @test 0 < elbo_vars.var_G.v[1] < 1e-2 * elbo_vars.E_G.v[1]^2
end


function test_that_star_truth_is_most_likely()
    blob, ea, body = true_star_init();
    best = DeterministicVI.elbo_likelihood(ea);

    for bad_a in [.3, .5, .9]
        ea_a = deepcopy(ea)
        ea_a.vp[1][ids.a[:, 1]] = [ 1.0 - bad_a, bad_a ]
        bad_a_lik = DeterministicVI.elbo_likelihood(ea_a)
        @test best.v[1] > bad_a_lik.v[1]
    end

    for h2 in -2:2
        for w2 in -2:2
            if !(h2 == 0 && w2 == 0)
                ea_mu = deepcopy(ea)
                ea_mu.vp[1][ids.u] += [h2 * .5, w2 * .5]
                bad_mu = DeterministicVI.elbo_likelihood(ea_mu)
                @test best.v[1] > bad_mu.v[1]
            end
        end
    end

    for delta in [.7, .9, 1.1, 1.3]
        ea_r1 = deepcopy(ea)
        ea_r1.vp[1][ids.r1] += log(delta)
        bad_r1 = DeterministicVI.elbo_likelihood(ea_r1)
        @test best.v[1] > bad_r1.v[1]
    end

    for b in 1:4
        for delta in [-.3, .3]
            ea_c1 = deepcopy(ea)
            ea_c1.vp[1][ids.c1[b, 1]] += delta
            bad_c1 = DeterministicVI.elbo_likelihood(ea_c1)
            @test best.v[1] > bad_c1.v[1]
        end
    end

end


function test_that_galaxy_truth_is_most_likely()
    blob, ea, body = gen_sample_galaxy_dataset(perturb=false);
    ea.vp[1][ids.a[:, 1]] = [ 0.01, .99 ]
    best = DeterministicVI.elbo_likelihood(ea);

    for bad_a in [.3, .5, .9]
        ea_a = deepcopy(ea);
        ea_a.vp[1][ids.a[:, 1]] = [ 1.0 - bad_a, bad_a ];
        bad_a =
          DeterministicVI.elbo_likelihood(ea_a; calculate_derivs=false);
        @test best.v[1] > bad_a.v[1];
    end

    for h2 in -2:2
        for w2 in -2:2
            if !(h2 == 0 && w2 == 0)
                ea_mu = deepcopy(ea)
                ea_mu.vp[1][ids.u] += [h2 * .5, w2 * .5]
                bad_mu = DeterministicVI.elbo_likelihood(
                  ea_mu; calculate_derivs=false)
                @test best.v[1] > bad_mu.v[1]
            end
        end
    end

    for bad_scale in [.8, 1.2]
        ea_r1 = deepcopy(ea)
        ea_r1.vp[1][ids.r1] += 2 * log(bad_scale)
        bad_r1 = DeterministicVI.elbo_likelihood(
          ea_r1; calculate_derivs=false)
        @test best.v[1] > bad_r1.v[1]
    end

    for n in [:e_axis, :e_angle, :e_scale]
        for bad_scale in [.8, 1.2]
            ea_bad = deepcopy(ea)
            ea_bad.vp[1][getfield(ids, n)] *= bad_scale
            bad_elbo = DeterministicVI.elbo_likelihood(
              ea_bad; calculate_derivs=false)
            @test best.v[1] > bad_elbo.v[1]
        end
    end

    for b in 1:4
        for delta in [-.3, .3]
            ea_c1 = deepcopy(ea)
            ea_c1.vp[1][ids.c1[b, 2]] += delta
            bad_c1 = DeterministicVI.elbo_likelihood(
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
      [ [ WCS.world_to_pix(blob[b].wcs, ce.pos) for b=1:5 ]
        for ce in cat_entries ]

    function ce_inbounds(ce)
        pix_locs = [ WCS.world_to_pix(blob[b].wcs, ce.pos) for b=1:5 ]
        inbounds(pos) = pos[1] > -10. && pos[2] > -10 &&
                        pos[1] < 61 && pos[2] < 61
        reduce(&, [inbounds(pos) for pos in pix_locs])
    end
    cat_entries = filter(ce_inbounds, cat_entries)

    ea = make_elbo_args(blob, cat_entries)
    for s in 1:length(cat_entries)
        ea.vp[s][ids.a[2, 1]] = cat_entries[s].is_star ? 0.01 : 0.99
        ea.vp[s][ids.a[1, 1]] = 1.0 - ea.vp[s][ids.a[2, 1]]
    end
    best = DeterministicVI.elbo_likelihood(ea; calculate_derivs=false);

    # s is the brightest source.
    s = 1

    for bad_scale in [.7, 1.3]
        ea_r1 = deepcopy(ea)
        ea_r1.vp[s][ids.r1] += 2 * log(bad_scale)
        bad_r1 = DeterministicVI.elbo_likelihood(
          ea_r1; calculate_derivs=false)
        @test best.v[1] > bad_r1.v[1]
    end

    for n in [:e_axis, :e_angle, :e_scale]
        for bad_scale in [.6, 1.8]
            ea_bad = deepcopy(ea)
            ea_bad.vp[s][getfield(ids, n)] *= bad_scale
            bad_elbo = DeterministicVI.elbo_likelihood(
              ea_bad; calculate_derivs=false)
            @test best.v[1] > bad_elbo.v[1]
        end
    end

    for bad_a in [.3, .7]
        ea_a = deepcopy(ea)
        ea_a.vp[s][ids.a[:, 1]] = [ 1.0 - bad_a, bad_a ]

        bad_a = DeterministicVI.elbo_likelihood(
          ea_a; calculate_derivs=false)
        @test best.v[1] > bad_a.v[1]
    end

    for h2 in -2:2
        for w2 in -2:2
            if !(h2 == 0 && w2 == 0)
                ea_mu = deepcopy(ea)
                ea_mu.vp[s][ids.u] += [0.5h2, 0.5w2]
                bad_mu = DeterministicVI.elbo_likelihood(
                  ea_mu; calculate_derivs=false)
                @test best.v[1] > bad_mu.v[1]
            end
        end
    end

    for b in 1:4
        for delta in [-2., 2.]
            ea_c1 = deepcopy(ea)
            ea_c1.vp[s][ids.c1[b, :]] += delta
            bad_c1 = DeterministicVI.elbo_likelihood(
              ea_c1; calculate_derivs=false)
            info("$(best.v[1])  >  $(bad_c1.v[1])")
            @test best.v[1] > bad_c1.v[1]
        end
    end
end


function test_tiny_image_tiling()
  # Test that the tiling doesn't matter much for a body that is nearly a
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
    [img], catalog, patch_radius_pix=Inf)

  elbo_lik = DeterministicVI.elbo_likelihood(ea0;
        calculate_derivs=false, calculate_hessian=false);

  tile_width = 2
  ea1 = make_elbo_args(
    [img], catalog, tile_width=tile_width, patch_radius_pix=10.);
  elbo_lik_tiles =
    DeterministicVI.elbo_likelihood(
      ea1, calculate_derivs=false, calculate_hessian=false);

  tile_width = 5
  ea2 = make_elbo_args(
      [img], catalog, tile_width=tile_width, patch_radius_pix=10.);
  elbo_lik_tiles2 =
    DeterministicVI.elbo_likelihood(
      ea2, calculate_derivs=false, calculate_hessian=false);

  @test_approx_eq elbo_lik_tiles.v[1] elbo_lik_tiles2.v[1]
  @test_approx_eq_eps elbo_lik.v[1] elbo_lik_tiles.v[1] 100.

end


function test_num_allowed_sd()
    blob, ea, body = gen_two_body_dataset()

    ea.num_allowed_sd = Inf
    elbo_inf = DeterministicVI.elbo(ea)

    ea.num_allowed_sd = 3
    elbo_4sd = DeterministicVI.elbo(ea)

    @test_approx_eq elbo_inf.v[1] elbo_4sd.v[1]
    @test_approx_eq elbo_inf.d elbo_4sd.d
    @test_approx_eq elbo_inf.h elbo_4sd.h
end


####################################################

test_set_hess()
test_bvn_cov()
test_tile_predicted_image()
test_derivative_flags()
#test_active_sources()
test_tiny_image_tiling()
test_that_variance_is_low()
test_that_star_truth_is_most_likely()
test_that_galaxy_truth_is_most_likely()
test_coadd_cat_init_is_most_likely()
