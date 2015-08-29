using Celeste
using CelesteTypes
using Base.Test
using Distributions
using SampleData

import SloanDigitalSkySurvey: SDSS
import SloanDigitalSkySurvey: WCS

println("Running ELBO value tests.")


function true_star_init()
    blob, mp, body = gen_sample_star_dataset(perturb=false)

    mp.vp[1][ids.a] = [ 1.0 - 1e-4, 1e-4 ]
    mp.vp[1][ids.r2] = 1e-4
    mp.vp[1][ids.r1] = sample_star_fluxes[3] ./ mp.vp[1][ids.r2]
    mp.vp[1][ids.c2] = 1e-4

    blob, mp, body
end

#################################

function test_kl_divergence_values()
    blob, mp, three_bodies = gen_three_body_dataset()

    s = 1
    i = 1
    d = 1
    sample_size = 2_000_000

    function test_kl(q_dist, p_dist, subtract_kl_fun!)
        q_samples = rand(q_dist, sample_size)
        empirical_kl_samples = logpdf(q_dist, q_samples) - logpdf(p_dist, q_samples)
        empirical_kl = mean(empirical_kl_samples)
        accum = zero_sensitive_float(CanonicalParams)
        subtract_kl_fun!(accum)
        exact_kl = -accum.v
        tol = 4 * std(empirical_kl_samples) / sqrt(sample_size)
        min_diff = 1e-2 * std(empirical_kl_samples) / sqrt(sample_size)
        @test_approx_eq_eps empirical_kl exact_kl tol
    end

    vs = mp.vp[s]

    # a
    q_a = Bernoulli(vs[ids.a[2]])
    p_a = Bernoulli(mp.pp.a[2])
    test_kl(q_a, p_a, (accum) -> ElboDeriv.subtract_kl_a!(s, mp, accum))

    # k
    q_k = Categorical(vs[ids.k[:, i]])
    p_k = Categorical(mp.pp.k[i])
    function sklk(accum)
        ElboDeriv.subtract_kl_k!(i, s, mp, accum)
        @assert i == 1
        accum.v /= vs[ids.a[i]]
    end
    test_kl(q_k, p_k, sklk)

    # c
    mp.pp.c[i][1][:, d] = vs[ids.c1[:, i]]
    mp.pp.c[i][2][:, :, d] = diagm(vs[ids.c2[:, i]])
    q_c = MvNormal(vs[ids.c1[:, i]], diagm(vs[ids.c2[:, i]]))
    p_c = MvNormal(mp.pp.c[i][1][:, d], mp.pp.c[i][2][:, :, d])
    function sklc(accum)
        ElboDeriv.subtract_kl_c!(d, i, s, mp, accum)
        accum.v /= vs[ids.a[i]] * vs[ids.k[d, i]]
    end
    test_kl(q_c, p_c, sklc)

    # r
    q_r = Gamma(vs[ids.r1[i]], vs[ids.r2[i]])
    p_r = Gamma(mp.pp.r[i][1], mp.pp.r[i][2])
    function sklr(accum)
        ElboDeriv.subtract_kl_r!(i, s, mp, accum)
        @assert i == 1
        accum.v /= vs[ids.a[i]]
    end
    test_kl(q_r, p_r, sklr)

end


function test_that_variance_is_low()
    # very peaked variational distribution---variance for F(m) should be low
    blob, mp, body = true_star_init()

    test_b = 3
    set_patch_wcs!(mp.patches[1], blob[test_b].wcs)
    star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(blob[test_b].psf, mp, test_b)
    fs0m = zero_sensitive_float(StarPosParams)
    fs1m = zero_sensitive_float(GalaxyPosParams)
    E_G = zero_sensitive_float(CanonicalParams)
    var_G = zero_sensitive_float(CanonicalParams)
    sb = ElboDeriv.SourceBrightness(mp.vp[1])
    m_pos = Float64[10, 12]
    wcs_jacobian = WCS.pixel_world_jacobian(blob[test_b].wcs, m_pos)
    ElboDeriv.accum_pixel_source_stats!(sb, star_mcs, gal_mcs,
        mp.vp[1], 1, 1, m_pos, 3, fs0m, fs1m, E_G, var_G, wcs_jacobian)

    @test 0 < var_G.v < 1e-2 * E_G.v^2
end


function test_that_star_truth_is_most_likely()
    blob, mp, body = true_star_init()

    best = ElboDeriv.elbo_likelihood(blob, mp)

    for bad_a in [.3, .5, .9]
        mp_a = deepcopy(mp)
        mp_a.vp[1][ids.a] = [ 1.0 - bad_a, bad_a ]
        bad_a_lik = ElboDeriv.elbo_likelihood(blob, mp_a)
        @test best.v > bad_a_lik.v
    end

    for h2 in -2:2
        for w2 in -2:2
            if !(h2 == 0 && w2 == 0)
                mp_mu = deepcopy(mp)
                mp_mu.vp[1][ids.u] += [h2 * .5, w2 * .5]
                bad_mu = ElboDeriv.elbo_likelihood(blob, mp_mu)
                @test best.v > bad_mu.v
            end
        end
    end

    for delta in [.7, .9, 1.1, 1.3]
        mp_r1 = deepcopy(mp)
        mp_r1.vp[1][ids.r1] *= delta
        bad_r1 = ElboDeriv.elbo_likelihood(blob, mp_r1)
        @test best.v > bad_r1.v
    end

    for b in 1:4
        for delta in [-.3, .3]
            mp_c1 = deepcopy(mp)
            mp_c1.vp[1][ids.c1[b, 1]] += delta
            bad_c1 = ElboDeriv.elbo_likelihood(blob, mp_c1)
            @test best.v > bad_c1.v
        end
    end
end


function test_that_galaxy_truth_is_most_likely()
    blob, mp, body = gen_sample_galaxy_dataset(perturb=false)
    mp.vp[1][ids.a] = [ 0.01, .99 ]
    best = ElboDeriv.elbo_likelihood(blob, mp)

    for bad_a in [.3, .5, .9]
        mp_a = deepcopy(mp)
        mp_a.vp[1][ids.a] = [ 1.0 - bad_a, bad_a ]
        bad_a = ElboDeriv.elbo_likelihood(blob, mp_a)
        @test best.v > bad_a.v
    end

    for h2 in -2:2
        for w2 in -2:2
            if !(h2 == 0 && w2 == 0)
                mp_mu = deepcopy(mp)
                mp_mu.vp[1][ids.u] += [h2 * .5, w2 * .5]
                bad_mu = ElboDeriv.elbo_likelihood(blob, mp_mu)
                @test best.v > bad_mu.v
            end
        end
    end

    for bad_scale in [.8, 1.2]
        mp_r1 = deepcopy(mp)
        mp_r1.vp[1][ids.r1] *= bad_scale^2
        mp_r1.vp[1][ids.r2] /= bad_scale  # keep variance the same
        bad_r1 = ElboDeriv.elbo_likelihood(blob, mp_r1)
        @test best.v > bad_r1.v
    end

    for n in [:e_axis, :e_angle, :e_scale]
        for bad_scale in [.8, 1.2]
            mp_bad = deepcopy(mp)
            mp_bad.vp[1][ids.(n)] *= bad_scale
            bad_elbo = ElboDeriv.elbo_likelihood(blob, mp_bad)
            @test best.v > bad_elbo.v
        end
    end

    for b in 1:4
        for delta in [-.3, .3]
            mp_c1 = deepcopy(mp)
            mp_c1.vp[1][ids.c1[b, 2]] += delta
            bad_c1 = ElboDeriv.elbo_likelihood(blob, mp_c1)
            @test best.v > bad_c1.v
        end
    end
end


function test_coadd_cat_init_is_most_likely()  # on a real stamp
    # TODO: not currently passing.

    stamp_id = "5.0073-0.0739"
    blob = Images.load_stamp_blob(dat_dir, stamp_id)

    cat_entries = Images.load_stamp_catalog(dat_dir, "s82-$stamp_id", blob)
    bright(ce) = sum(ce.star_fluxes) > 3 || sum(ce.gal_fluxes) > 3
    cat_entries = filter(bright, cat_entries)

    ce_pix_locs = [ [ WCS.world_to_pixel(blob[b].wcs, ce.pos) for b=1:5 ] for ce in cat_entries ]

    function ce_inbounds(ce)
        pix_locs = [ WCS.world_to_pixel(blob[b].wcs, ce.pos) for b=1:5 ]
        inbounds(pos) = pos[1] > -10. && pos[2] > -10 &&
                        pos[1] < 61 && pos[2] < 61
        all([ inbounds(pos) for pos in pix_locs ])
    end
    cat_entries = filter(ce_inbounds, cat_entries)

    mp = ModelInit.cat_init(cat_entries)
    for s in 1:length(cat_entries)
        mp.vp[s][ids.a[2]] = cat_entries[s].is_star ? 0.01 : 0.99
        mp.vp[s][ids.a[1]] = 1.0 - mp.vp[s][ids.a[2]]
    end
    best = ElboDeriv.elbo_likelihood(blob, mp)

    # s is the brightest source.
    s = 1

    for bad_scale in [.7, 1.3]
        mp_r1 = deepcopy(mp)
        mp_r1.vp[s][ids.r1] *= bad_scale^2
        mp_r1.vp[s][ids.r2] /= bad_scale  # keep variance the same
        bad_r1 = ElboDeriv.elbo_likelihood(blob, mp_r1)
        @test best.v > bad_r1.v
    end

    for n in [:e_axis, :e_angle, :e_scale]
        for bad_scale in [.6, 1.8]
            mp_bad = deepcopy(mp)
            mp_bad.vp[s][ids.(n)] *= bad_scale
            bad_elbo = ElboDeriv.elbo_likelihood(blob, mp_bad)
            @test best.v > bad_elbo.v
        end
    end

    for bad_a in [.3, .7]
        mp_a = deepcopy(mp)
        mp_a.vp[s][ids.a] = [ 1.0 - bad_a, bad_a ]

        bad_a = ElboDeriv.elbo_likelihood(blob, mp_a)
        @test best.v > bad_a.v
    end

    for h2 in -2:2
        for w2 in -2:2
            if !(h2 == 0 && w2 == 0)
                mp_mu = deepcopy(mp)
                mp_mu.vp[s][ids.u] += [0.5h2, 0.5w2]
                bad_mu = ElboDeriv.elbo_likelihood(blob, mp_mu)
                @test best.v > bad_mu.v
            end
        end
    end

    for b in 1:4
        for delta in [-2., 2.]
            mp_c1 = deepcopy(mp)
            mp_c1.vp[s][ids.c1[b, :]] += delta
            bad_c1 = ElboDeriv.elbo_likelihood(blob, mp_c1)
            info("$(best.v)  >  $(bad_c1.v)")
            @test best.v > bad_c1.v
        end
    end
end


function test_tiny_image_tiling()
    blob0 = Images.load_stamp_blob(dat_dir, "164.4311-39.0359")
    pc = PsfComponent(1./3, zeros(2), 1e-4 * eye(2))
    trivial_psf = [pc, pc, pc]
    pixels = ones(100, 1) * 12
    pixels[98:100, 1] = [1e3, 1e4, 1e5]
    img = Image(3, 1, pixels, 3, blob0[3].wcs, 3., 4., trivial_psf, 1, 1, 1)
    catalog = [sample_ce([100., 1], true),]
    catalog[1].star_fluxes = ones(5) * 1e5

    mp0 = ModelInit.cat_init(catalog)
    accum0 = zero_sensitive_float(CanonicalParams)
    ElboDeriv.elbo_likelihood!(img, mp0, accum0)

    mp_tiles = ModelInit.cat_init(catalog, patch_radius=10., tile_width=2)
    accum_tiles = zero_sensitive_float(CanonicalParams)
    ElboDeriv.elbo_likelihood!(img, mp_tiles, accum_tiles)

    mp_tiles2 = ModelInit.cat_init(catalog, patch_radius=10., tile_width=5)
    accum_tiles2 = zero_sensitive_float(CanonicalParams)
    ElboDeriv.elbo_likelihood!(img, mp_tiles, accum_tiles2)
    @test_approx_eq accum_tiles.v accum_tiles2.v

    @test_approx_eq_eps accum0.v accum_tiles.v 100.
end

#################

function test_elbo_with_nan()
    blob, mp, body = gen_sample_star_dataset(perturb=false)

    # Set to 5 to test the code for tiles with no sources.
    mp.tile_width = 5
    initial_elbo = ElboDeriv.elbo(blob, mp)

    for b in 1:5
        blob[b].pixels[1,1] = NaN
    end

    nan_elbo = ElboDeriv.elbo(blob, mp)

    # We deleted a pixel, so there's reason to expect them to be different,
    # but importantly they're reasonably close and not NaN.
    @test_approx_eq_eps (nan_elbo.v - initial_elbo.v) / initial_elbo.v 0. 1e-4
    deriv_rel_err = (nan_elbo.d - initial_elbo.d) ./ initial_elbo.d
    @test_approx_eq_eps deriv_rel_err fill(0., length(mp.vp[1])) 0.05
end


####################################################

test_kl_divergence_values()
test_that_variance_is_low()
test_that_star_truth_is_most_likely()
test_that_galaxy_truth_is_most_likely()
test_coadd_cat_init_is_most_likely()
test_tiny_image_tiling()
test_elbo_with_nan()
