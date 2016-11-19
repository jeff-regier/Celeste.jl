using Celeste: DeterministicVI, SensitiveFloats

using Base.Test
using Distributions
using DerivativeTestUtils
using StaticArrays


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


function test_derivative_flags()
    images, ea, body = gen_two_body_dataset()

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

    images, ea, bodies = gen_two_body_dataset()

    s = 1
    n = 5
    p = ea.patches[s, n]

    # for subsequent tests, ensure that the second light source
    # has a radius at least as large as both the height and the
    # width of image 3, and that its center is within the image
    @test p.radius_pix >= 23
    @test 0.5 <= p.center[1] <= 20.5
    @test 0.5 <= p.center[2] <= 23.5

    # this patch is huge, the bottom left corner should be the
    # bottom left of the image
    @test p.bitmap_corner == [1, 1]

    # this patch is huge, all the pixels should be active pixels
    @test size(p.active_pixel_bitmap) == size(images[n].pixels)
    @test all(p.active_pixel_bitmap)

    # lets make the second source have only a few active pixels
    # in one band, so it and source 1 don't overlap completely
    s2 = 2
    p2 = ea.patches[s2, n]
    fill!(p2.active_pixel_bitmap, false)
    p2.active_pixel_bitmap[10:11,10:11] = true

    # now on to the main test
    ea12 = ElboArgs(ea.images, ea.vp, ea.patches, [1, 2], [1, 2])
    elbo_lik_12 = DeterministicVI.elbo_likelihood(ea12)

    ea1 = ElboArgs(ea.images, ea.vp, ea.patches, [1,], [1, 2])
    elbo_lik_1 = DeterministicVI.elbo_likelihood(ea1)

    ea2 = ElboArgs(ea.images, ea.vp, ea.patches, [2,], [1, 2])
    elbo_lik_2 = DeterministicVI.elbo_likelihood(ea2)

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
    images, ea, body = true_star_init()
    n = 1

    star_mcs, gal_mcs = Model.load_bvn_mixtures(ea.S, ea.patches,
                                ea.vp, ea.active_sources,
                                ea.psf_K, n)
    sbs = DeterministicVI.SourceBrightness{Float64}[
        DeterministicVI.SourceBrightness(ea.vp[s]) for s in 1:ea.S]

    elbo_vars = DeterministicVI.ElboIntermediateVariables(
      Float64, ea.S, length(ea.active_sources))

    clear!(elbo_vars.E_G)
    clear!(elbo_vars.var_G)

    h, w = 10, 12
    DeterministicVI.get_expected_pixel_brightness!(
      elbo_vars, n, h, w, sbs, star_mcs, gal_mcs, ea)

    @test 0 < elbo_vars.var_G.v[1] < 1e-2 * elbo_vars.E_G.v[1]^2
end


function test_that_star_truth_is_most_likely()
    images, ea, body = true_star_init()
    best = DeterministicVI.elbo_likelihood(ea)

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
    images, ea, body = gen_sample_galaxy_dataset(perturb=false)
    ea.vp[1][ids.a[:, 1]] = [ 0.01, .99 ]
    best = DeterministicVI.elbo_likelihood(ea)

    for bad_a in [.3, .5, .9]
        ea_a = deepcopy(ea)
        ea_a.vp[1][ids.a[:, 1]] = [ 1.0 - bad_a, bad_a ]
        bad_a =
          DeterministicVI.elbo_likelihood(ea_a; calculate_derivs=false)
        @test best.v[1] > bad_a.v[1]
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
    images = SampleData.load_stamp_blob(datadir, stamp_id)

    cat_entries = SampleData.load_stamp_catalog(datadir, "s82-$stamp_id", images)
    bright(ce) = sum(ce.star_fluxes) > 3 || sum(ce.gal_fluxes) > 3
    cat_entries = filter(bright, cat_entries)

    ce_pix_locs =
      [ [ WCS.world_to_pix(images[b].wcs, ce.pos) for b=1:5 ]
        for ce in cat_entries ]

    function ce_inbounds(ce)
        pix_locs = [ WCS.world_to_pix(images[b].wcs, ce.pos) for b=1:5 ]
        inbounds(pos) = pos[1] > -10. && pos[2] > -10 &&
                        pos[1] < 61 && pos[2] < 61
        reduce(&, [inbounds(pos) for pos in pix_locs])
    end
    cat_entries = filter(ce_inbounds, cat_entries)

    ea = make_elbo_args(images, cat_entries)
    for s in 1:length(cat_entries)
        ea.vp[s][ids.a[2, 1]] = cat_entries[s].is_star ? 0.01 : 0.99
        ea.vp[s][ids.a[1, 1]] = 1.0 - ea.vp[s][ids.a[2, 1]]
    end
    best = DeterministicVI.elbo_likelihood(ea; calculate_derivs=false)

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


function test_num_allowed_sd()
    images, ea, body = gen_two_body_dataset()

    ea.num_allowed_sd = Inf
    elbo_inf = DeterministicVI.elbo(ea)

    ea.num_allowed_sd = 3
    elbo_4sd = DeterministicVI.elbo(ea)

    @test_approx_eq elbo_inf.v[1] elbo_4sd.v[1]
    @test_approx_eq elbo_inf.d elbo_4sd.d
    @test_approx_eq elbo_inf.h elbo_4sd.h
end


function test_populate_fsm!()
    images, ea, body = gen_two_body_dataset()

    n = 3
    h = w = 5
    s = 2
    ea.active_sources = [s]

    star_mcs, gal_mcs = Model.load_bvn_mixtures(ea.S, ea.patches,
                                ea.vp, ea.active_sources,
                                ea.psf_K, n)
    Model.populate_fsm_vecs!(ea.elbo_vars.bvn_derivs,
                             ea.elbo_vars.fs0m_vec,
                             ea.elbo_vars.fs1m_vec,
                             ea.elbo_vars.calculate_derivs,
                             ea.elbo_vars.calculate_hessian,
                             ea.patches,
                             ea.active_sources,
                             ea.num_allowed_sd,
                             n, h, w,
                             gal_mcs, star_mcs)

    fs0m = zero_sensitive_float(StarPosParams, Float64)
    fs1m = zero_sensitive_float(GalaxyPosParams, Float64)

    x = @SVector Float64[h, w]
    elbo_vars = ea.elbo_vars
    Model.populate_fsm!(elbo_vars.bvn_derivs,
                        fs0m, fs1m,
                        elbo_vars.calculate_derivs,
                        elbo_vars.calculate_hessian,
                        s, x, true,
                        ea.num_allowed_sd,
                        ea.patches[s, n].wcs_jacobian,
                        gal_mcs, star_mcs)

    @test_approx_eq fs0m.v[1] elbo_vars.fs0m_vec[s].v[1]
    @test_approx_eq fs0m.d elbo_vars.fs0m_vec[s].d
    @test_approx_eq fs0m.h elbo_vars.fs0m_vec[s].h

    @test_approx_eq fs1m.v[1] elbo_vars.fs1m_vec[s].v[1]
    @test_approx_eq fs1m.d elbo_vars.fs1m_vec[s].d
    @test_approx_eq fs1m.h elbo_vars.fs1m_vec[s].h
end


test_active_sources()
test_set_hess()
test_bvn_cov()
test_derivative_flags()
test_num_allowed_sd()
#test_that_variance_is_low()
test_that_star_truth_is_most_likely()
test_that_galaxy_truth_is_most_likely()
test_coadd_cat_init_is_most_likely()
test_populate_fsm!()
