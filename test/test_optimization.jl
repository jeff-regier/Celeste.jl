
function verify_sample_star(vs, pos)
    @test_approx_eq vs[ids.chi] 0.01

    @test_approx_eq_eps vs[ids.mu[1]] pos[1] 0.1
    @test_approx_eq_eps vs[ids.mu[2]] pos[2] 0.1

    brightness_hat = vs[ids.gamma[1]] * vs[ids.zeta[1]]
    @test_approx_eq_eps brightness_hat / sample_star_fluxes[3] 1. 0.01

    true_colors = log(sample_star_fluxes[2:5] ./ sample_star_fluxes[1:4])
    for b in 1:4
        @test_approx_eq_eps vs[ids.beta[b, 1]] true_colors[b] 0.2
    end
end

function verify_sample_galaxy(vs, pos)
    @test_approx_eq vs[ids.chi] 0.99

    @test_approx_eq_eps vs[ids.mu[1]] pos[1] 0.1
    @test_approx_eq_eps vs[ids.mu[2]] pos[2] 0.1

    @test_approx_eq_eps vs[ids.rho] .7 0.05
    @test_approx_eq_eps vs[ids.theta] 0.1 0.08
    @test_approx_eq_eps vs[ids.sigma] 4. 0.2

    phi_hat = vs[ids.phi]
    phi_hat -= floor(phi_hat / pi) * pi
    five_deg = 5 * pi/180
    @test_approx_eq_eps phi_hat pi/4 five_deg

    brightness_hat = vs[ids.gamma[2]] * vs[ids.zeta[2]]
    @test_approx_eq_eps brightness_hat / sample_galaxy_fluxes[3] 1. 0.01

    true_colors = log(sample_galaxy_fluxes[2:5] ./ sample_galaxy_fluxes[1:4])
    for b in 1:4
        @test_approx_eq_eps vs[ids.beta[b, 2]] true_colors[b] 0.2
    end
end


#########################################################


function test_star_optimization()
    blob, mp, body = gen_sample_star_dataset()
    OptimizeElbo.maximize_likelihood(blob, mp)
    verify_sample_star(mp.vp[1], [10.1, 12.2])
end


function test_galaxy_optimization()
    blob, mp, body = gen_sample_galaxy_dataset()
    OptimizeElbo.maximize_likelihood(blob, mp)
    verify_sample_galaxy(mp.vp[1], [8.5, 9.6])
end


function test_kappa_finding()
    blob, mp, body = gen_sample_galaxy_dataset()
    omitted_ids = setdiff(all_params, ids.kappa[:])

    get_kl_gal_c() = begin
        accum = zero_sensitive_float([1], all_params)
        for d in 1:D
            ElboDeriv.subtract_kl_c!(d, 2, 1, mp, accum)
        end
        -accum.v
    end

    mp.vp[1][ids.kappa[:, 2]] = [0.01, 0.99]
    mp.vp[1][ids.beta[:,2]] = mp.pp.Omega[2][:, 2]
    lower_klc = get_kl_gal_c()
    mp.vp[1][ids.beta[:,2]] = mp.pp.Omega[2][:, 1]
    higher_klc = get_kl_gal_c()
    @test lower_klc < higher_klc
    println(lower_klc, "  ", higher_klc)

    mp.vp[1][ids.kappa[:, 2]] = [0.99, 0.01]
    mp.vp[1][ids.beta[:,2]] = mp.pp.Omega[2][:, 1]
    lower_klc = get_kl_gal_c()
    mp.vp[1][ids.beta[:,2]] = mp.pp.Omega[2][:, 2]
    higher_klc = get_kl_gal_c()
    println(lower_klc, "  ", higher_klc)
    @test lower_klc < higher_klc

    mp.pp.Lambda[2][1][:, :] = mp.pp.Lambda[2][2][:, :] = eye(4)
    klc_wrapper(blob, mp) = begin
        accum = zero_sensitive_float([1], all_params)
        for d in 1:D
            ElboDeriv.subtract_kl_c!(d, 2, 1, mp, accum)
        end
        accum
    end

    mp.vp[1][ids.beta[:,2]] = mp.pp.Omega[2][:, 1]
    mp.vp[1][ids.kappa[:, 2]] = [0.5, 0.5]
    OptimizeElbo.maximize_f(klc_wrapper, blob, mp, omitted_ids=omitted_ids)
    @test mp.vp[1][ids.kappa[1, 2]] > .9

    mp.vp[1][ids.beta[:,2]] = mp.pp.Omega[2][:, 2]
    mp.vp[1][ids.kappa[:, 2]] = [0.5, 0.5]
    OptimizeElbo.maximize_f(klc_wrapper, blob, mp, omitted_ids=omitted_ids)
    @test mp.vp[1][ids.kappa[2, 2]] > .9

    mp.pp.Xi[2] = [.9, .1]
    mp.vp[1][ids.beta[:,2]] = mp.pp.Omega[2][:, 1]
    mp.vp[1][ids.kappa[:, 2]] = [0.5, 0.5]
    OptimizeElbo.maximize_f(ElboDeriv.elbo, blob, mp, omitted_ids=omitted_ids)
    @test mp.vp[1][ids.kappa[1, 2]] > .9

    mp.pp.Xi[2] = [.1, .9]
    mp.vp[1][ids.beta[:,2]] = mp.pp.Omega[2][:, 2]
    mp.vp[1][ids.kappa[:, 2]] = [0.5, 0.5]
    OptimizeElbo.maximize_f(ElboDeriv.elbo, blob, mp, omitted_ids=omitted_ids)
    @test mp.vp[1][ids.kappa[2, 2]] > .9
end


function test_bad_chi_init()
    gal_color_mode = [ 2.47122, 1.832, 4.0, 5.9192, 9.12822]
    ce = CatalogEntry([7.2, 8.3], false, gal_color_mode, gal_color_mode,
            0.5, .7, pi/4, .5)

    blob0 = SDSS.load_stamp_blob(dat_dir, "164.4311-39.0359")
    for b in 1:5
        blob0[b].H, blob0[b].W = 20, 23
    end
    blob = Synthetic.gen_blob(blob0, [ce,])

    mp = ModelInit.cat_init([ce,])
    mp.vp[1][ids.chi] = 0.5
    omitted_ids = [ids.chi]
    OptimizeElbo.maximize_f(ElboDeriv.elbo, blob, mp, omitted_ids=omitted_ids)

    mp.vp[1][ids.chi] = 0.2
    elbo_bad = ElboDeriv.elbo_likelihood(blob, mp)
    @test elbo_bad.d[ids.chi, 1] > 0

    for i = 0:7
        mp.vp[1][ids.chi] = 0.2 + .1i
        el_bad = ElboDeriv.elbo_likelihood(blob, mp)
        elbo_bad = ElboDeriv.elbo(blob, mp)
        println("like: ", mp.vp[1][ids.chi], "  ", el_bad.v, "  ", el_bad.d[ids.chi, 1])
        println("elbo", mp.vp[1][ids.chi], "  ", elbo_bad.v, "  ", elbo_bad.d[ids.chi, 1])
    end

    omitted_ids = setdiff(all_params, [ids.chi])
    OptimizeElbo.maximize_f(ElboDeriv.elbo, blob, mp, omitted_ids=omitted_ids)
    @test mp.vp[1][ids.chi] >= 0.5

    mp2 = deepcopy(mp)
    mp2.vp[1][ids.chi] = 0.99
    elbo_true2 = ElboDeriv.elbo_likelihood(blob, mp2)
    mp2.vp[1][ids.chi] = 0.01
    elbo_bad2 = ElboDeriv.elbo_likelihood(blob, mp2)
    @test elbo_true2.v > elbo_bad2.v
    @test elbo_bad2.d[ids.chi, 1] > 0
end


function test_likelihood_invariance_to_chi()
    fluxes = [2.47122, 1.832, 4.0, 5.9192, 9.12822]
    ce = CatalogEntry([7.2,8.3], false, fluxes, fluxes, 0.5, .7, pi/4, .5)

    blob0 = SDSS.load_stamp_blob(dat_dir, "164.4311-39.0359")
    for b in 1:5
        blob0[b].H, blob0[b].W = 20, 23
    end
    blob = Synthetic.gen_blob(blob0, [ce,])

    mp = ModelInit.cat_init([ce,])
    mp.vp[1][ids.chi] = 0.2
    omitted_ids = [ids.chi, ids.zeta[:]]
    OptimizeElbo.maximize_f(ElboDeriv.elbo_likelihood, blob, mp, omitted_ids=omitted_ids)

    mp2 = ModelInit.cat_init([ce,])
    mp2.vp[1][ids.chi] = 0.8
    OptimizeElbo.maximize_f(ElboDeriv.elbo_likelihood, blob, mp2, omitted_ids=omitted_ids)

    mp.vp[1][ids.chi] = 0.5
    mp2.vp[1][ids.chi] = 0.5
    @test_approx_eq_eps(ElboDeriv.elbo_likelihood(blob, mp).v,
        ElboDeriv.elbo_likelihood(blob, mp2).v, 1)

    for i in 2:length(all_params) #skip chi
        println(mp.vp[1][i], "  vs  ",  mp2.vp[1][i])
        @test_approx_eq_eps mp.vp[1][i] / mp2.vp[1][i] 1. 0.1
    end
end


function test_kl_invariance_to_chi()
    fluxes = [2.47122, 1.832, 4.0, 5.9192, 9.12822]
    ce = CatalogEntry([7.2,8.3], false, fluxes, fluxes, 0.5, .7, pi/4, .5)
    blob0 = SDSS.load_stamp_blob(dat_dir, "164.4311-39.0359")
    for b in 1:5
        blob0[b].H, blob0[b].W = 20, 23
    end
    blob = Synthetic.gen_blob(blob0, [ce,])

    kl_wrapper(blob, mp) = begin
        accum = zero_sensitive_float([1], all_params)
        ElboDeriv.subtract_kl!(mp, accum)
        accum
    end

    mp = ModelInit.cat_init([ce,])
    mp.vp[1][ids.chi] = 0.8
    omitted_ids = [ids.chi]
    OptimizeElbo.maximize_f(kl_wrapper, blob, mp, omitted_ids=omitted_ids, ftol_abs=1e-9)

    mp2 = ModelInit.cat_init([ce,])
    mp2.vp[1][ids.chi] = 0.2
    OptimizeElbo.maximize_f(kl_wrapper, blob, mp2, omitted_ids=omitted_ids, ftol_abs=1e-9)

    mp.vp[1][ids.chi] = 0.5
    mp2.vp[1][ids.chi] = 0.5
    @test_approx_eq_eps kl_wrapper(blob, mp).v kl_wrapper(blob, mp2).v 1e-1

    for i in 2:length(all_params) #skip chi
        println(mp.vp[1][i], "  vs  ",  mp2.vp[1][i])
        @test_approx_eq_eps mp.vp[1][i] / mp2.vp[1][i] 1. 0.1
    end
end


function test_elbo_invariance_to_chi()
    fluxes = [2.47122, 1.832, 4.0, 5.9192, 9.12822] * 100
    ce = CatalogEntry([7.2,8.3], false, fluxes, fluxes, 0.5, .7, pi/4, .5)
    blob0 = SDSS.load_stamp_blob(dat_dir, "164.4311-39.0359")
    for b in 1:5
        blob0[b].H, blob0[b].W = 20, 23
    end
    blob = Synthetic.gen_blob(blob0, [ce,])

    mp = ModelInit.cat_init([ce,])
    mp.vp[1][ids.chi] = 0.2
    omitted_ids = [ids.chi, ids.zeta[:], ids.lambda[:], ids.theta]
    OptimizeElbo.maximize_f(ElboDeriv.elbo, blob, mp, omitted_ids=omitted_ids)

    mp2 = ModelInit.cat_init([ce,])
    mp2.vp[1][ids.chi] = 0.8
    OptimizeElbo.maximize_f(ElboDeriv.elbo, blob, mp2, omitted_ids=omitted_ids)

    mp.vp[1][ids.chi] = 0.5
    mp2.vp[1][ids.chi] = 0.5
    @test_approx_eq_eps ElboDeriv.elbo(blob, mp).v ElboDeriv.elbo(blob, mp2).v 1

    for i in setdiff(all_params, [ids.chi]) #skip chi
        println(mp.vp[1][i], "  vs  ",  mp2.vp[1][i])
        @test_approx_eq_eps mp.vp[1][i] / mp2.vp[1][i] 1. 0.1
    end
end


function test_peak_init_galaxy_optimization()
    blob, mp, body = gen_sample_galaxy_dataset()
    mp = ModelInit.peak_init(blob)
    OptimizeElbo.maximize_likelihood(blob, mp)
    verify_sample_galaxy(mp.vp[1], [8.5, 9.6])
end


function test_peak_init_2body_optimization()
    srand(1)
    blob0 = SDSS.load_stamp_blob(dat_dir, "164.4311-39.0359")

    two_bodies = [
        sample_ce([11.1, 21.2], true),
        sample_ce([15.3, 31.4], false),
    ]

    blob = Synthetic.gen_blob(blob0, two_bodies)
    mp = ModelInit.peak_init(blob) #one giant tile, giant patches
    @test mp.S == 2

    OptimizeElbo.maximize_likelihood(blob, mp)

    verify_sample_star(mp.vp[1], [11.1, 21.2])
    verify_sample_galaxy(mp.vp[2], [15.3, 31.4])
end


function test_full_elbo_optimization()
    blob, mp, body = gen_sample_galaxy_dataset(perturb=true)
    OptimizeElbo.maximize_elbo(blob, mp)
    verify_sample_galaxy(mp.vp[1], [8.5, 9.6])
end


function test_real_stamp_optimization()
    blob = SDSS.load_stamp_blob(dat_dir, "5.0073-0.0739")
    cat_entries = SDSS.load_stamp_catalog(dat_dir, "s82-5.0073-0.0739", blob)
    bright(ce) = sum(ce.star_fluxes) > 3 || sum(ce.gal_fluxes) > 3
    cat_entries = filter(bright, cat_entries)
    inbounds(ce) = ce.pos[1] > -10. && ce.pos[2] > -10 &&
        ce.pos[1] < 61 && ce.pos[2] < 61
    cat_entries = filter(inbounds, cat_entries)

    for ce in cat_entries
        println(ce)
    end

    mp = ModelInit.cat_init(cat_entries)
    OptimizeElbo.maximize_elbo(blob, mp)
end


function test_bad_galaxy_init()
    stamp_id = "5.0624-0.1528"
    blob0 = SDSS.load_stamp_blob(ENV["STAMP"], stamp_id)

    only_center(ce) = ce.pos[1] > 25. && ce.pos[2] > 25 &&
        ce.pos[1] < 27 && ce.pos[2] < 27

    cat_coadd = SDSS.load_stamp_catalog(ENV["STAMP"], "s82-$stamp_id", blob0)
    cat_coadd = filter(only_center, cat_coadd)
    @test length(cat_coadd) == 1

    blob = Synthetic.gen_blob(blob0, cat_coadd)

    cat_primary = SDSS.load_stamp_catalog(ENV["STAMP"], stamp_id, blob, match_blob=true)
    cat_primary = filter(only_center, cat_primary)
    @test length(cat_primary) == 1

    mp_bad_init = ModelInit.cat_init(cat_primary)
    OptimizeElbo.maximize_f(ElboDeriv.elbo, blob, mp_bad_init, ftol_abs=1e-7)
    @test mp_bad_init.vp[1][ids.chi] > .5

    mp_good_init = ModelInit.cat_init(cat_coadd)
    OptimizeElbo.maximize_elbo(blob, mp_good_init)
    @test mp_good_init.vp[1][ids.chi] > .5

    @test_approx_eq_eps mp_good_init.vp[1][ids.sigma] mp_bad_init.vp[1][ids.sigma] 0.2
    @test_approx_eq_eps mp_good_init.vp[1][ids.rho] mp_bad_init.vp[1][ids.rho] 0.2
    @test_approx_eq_eps mp_good_init.vp[1][ids.theta] mp_bad_init.vp[1][ids.theta] 0.2
    @test_approx_eq_eps mp_good_init.vp[1][ids.phi] mp_bad_init.vp[1][ids.phi] 0.2
end

####################################################

#test_bad_galaxy_init()
test_kappa_finding()
test_bad_chi_init()
test_elbo_invariance_to_chi()
test_kl_invariance_to_chi()
test_likelihood_invariance_to_chi()
test_star_optimization()
test_full_elbo_optimization()
test_galaxy_optimization()
test_real_stamp_optimization()  # long running
#test_peak_init_2body_optimization()
#test_peak_init_galaxy_optimization()
