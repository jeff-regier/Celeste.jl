using Celeste
using CelesteTypes
using Base.Test
using SampleData
using Transform

import OptimizeElbo

function test_objective_wrapper()
    omitted_ids = Int64[];
    kept_ids = setdiff(1:length(ids_free), omitted_ids);
    blob, mp, body = SampleData.gen_two_body_dataset();
    trans = get_mp_transform(mp, loc_width=1.0);

    wrapper = OptimizeElbo.ObjectiveWrapperFunctions(mp -> ElboDeriv.elbo(blob, mp),
        mp, trans, kept_ids, omitted_ids);

    x = trans.vp_to_vector(mp.vp, omitted_ids);
    elbo_result = trans.transform_sensitive_float(ElboDeriv.elbo(blob, mp), mp);
    elbo_grad = reduce(vcat, [ elbo_result.d[kept_ids, s] for s=1:mp.S ]);

    # Tese the print function
    wrapper.state.verbose = true
    wrapper.f_objective(x);
    wrapper.state.verbose = false

    w_v, w_grad = wrapper.f_value_grad(x);
    w_grad2 = zeros(Float64, length(x));
    wrapper.f_value_grad!(x, w_grad2);

    @test_approx_eq(w_v, elbo_result.v)
    @test_approx_eq(w_grad, elbo_grad)
    @test_approx_eq(w_grad, w_grad2)

    @test_approx_eq(w_v, wrapper.f_value(x))
    @test_approx_eq(w_grad, wrapper.f_grad(x))

    this_iter = wrapper.state.f_evals;
    wrapper.f_value(x);
    @test wrapper.state.f_evals == this_iter + 1

    # Test that the autodiff derivatives match the actual derivatives.
    w_ad_grad = wrapper.f_ad_grad(x);
    @test_approx_eq(w_grad, w_ad_grad)

    # Just test that the Hessian can be computed and is symmetric.
    w_hess = wrapper.f_ad_hessian(x);
    @test_approx_eq(w_hess, w_hess')
end


function verify_sample_star(vs, pos)
    @test vs[ids.a[2]] <= 0.01

    @test_approx_eq_eps vs[ids.u[1]] pos[1] 0.1
    @test_approx_eq_eps vs[ids.u[2]] pos[2] 0.1

    brightness_hat = vs[ids.r1[1]] * vs[ids.r2[1]]
    @test_approx_eq_eps brightness_hat / sample_star_fluxes[3] 1. 0.01

    true_colors = log(sample_star_fluxes[2:5] ./ sample_star_fluxes[1:4])
    for b in 1:4
        @test_approx_eq_eps vs[ids.c1[b, 1]] true_colors[b] 0.2
    end
end

function verify_sample_galaxy(vs, pos)
    @test vs[ids.a[2]] >= 0.99

    @test_approx_eq_eps vs[ids.u[1]] pos[1] 0.1
    @test_approx_eq_eps vs[ids.u[2]] pos[2] 0.1

    @test_approx_eq_eps vs[ids.e_axis] .7 0.05
    @test_approx_eq_eps vs[ids.e_dev] 0.1 0.08
    @test_approx_eq_eps vs[ids.e_scale] 4. 0.2

    phi_hat = vs[ids.e_angle]
    phi_hat -= floor(phi_hat / pi) * pi
    five_deg = 5 * pi/180
    @test_approx_eq_eps phi_hat pi/4 five_deg

    brightness_hat = vs[ids.r1[2]] * vs[ids.r2[2]]
    @test_approx_eq_eps brightness_hat / sample_galaxy_fluxes[3] 1. 0.01

    true_colors = log(sample_galaxy_fluxes[2:5] ./ sample_galaxy_fluxes[1:4])
    for b in 1:4
        @test_approx_eq_eps vs[ids.c1[b, 2]] true_colors[b] 0.2
    end
end


#########################################################


function test_star_optimization()
    blob, mp, body = gen_sample_star_dataset();
    trans = get_mp_transform(mp, loc_width=1.0);
    OptimizeElbo.maximize_likelihood(blob, mp, trans)
    verify_sample_star(mp.vp[1], [10.1, 12.2])
end


function test_galaxy_optimization()
    blob, mp, body = gen_sample_galaxy_dataset()
    trans = get_mp_transform(mp, loc_width=1.0);
    OptimizeElbo.maximize_likelihood(blob, mp, trans, xtol_rel=0.0)
    verify_sample_galaxy(mp.vp[1], [8.5, 9.6])
end


function test_kappa_finding()
    blob, mp, body = gen_sample_galaxy_dataset()
    trans = get_mp_transform(mp, loc_width=1.0);
    omitted_ids = setdiff(1:length(UnconstrainedParams), ids_free.k[:])

    get_kl_gal_c() = begin
        accum = zero_sensitive_float(CanonicalParams)
        for d in 1:D
            ElboDeriv.subtract_kl_c!(d, 2, 1, mp, accum)
        end
        -accum.v
    end

    mp.vp[1][ids.k[:, 2]] = [0.01, 0.99]
    mp.vp[1][ids.c1[:,2]] = mp.pp.c[2][1][:, 2]
    lower_klc = get_kl_gal_c()
    mp.vp[1][ids.c1[:,2]] = mp.pp.c[2][1][:, 1]
    higher_klc = get_kl_gal_c()
    @test lower_klc < higher_klc

    mp.vp[1][ids.k[:, 2]] = [0.99, 0.01]
    mp.vp[1][ids.c1[:,2]] = mp.pp.c[2][1][:, 1]
    lower_klc = get_kl_gal_c()
    mp.vp[1][ids.c1[:,2]] = mp.pp.c[2][1][:, 2]
    higher_klc = get_kl_gal_c()
    @test lower_klc < higher_klc

    mp.pp.c[2][2][:, :, 1] = mp.pp.c[2][2][:, :, 2] = eye(4)
    klc_wrapper(blob, mp) = begin
        accum = zero_sensitive_float(CanonicalParams)
        for d in 1:D
            ElboDeriv.subtract_kl_c!(d, 2, 1, mp, accum)
        end
        accum
    end

    mp.vp[1][ids.c1[:,2]] = mp.pp.c[2][1][:, 1]
    mp.vp[1][ids.k[:, 2]] = [0.5, 0.5]
    OptimizeElbo.maximize_f(klc_wrapper, blob, mp, trans, omitted_ids=omitted_ids)
    @test mp.vp[1][ids.k[1, 2]] > .9

    mp.vp[1][ids.c1[:,2]] = mp.pp.c[2][1][:, 2]
    mp.vp[1][ids.k[:, 2]] = [0.5, 0.5]
    OptimizeElbo.maximize_f(klc_wrapper, blob, mp, trans, omitted_ids=omitted_ids)
    @test mp.vp[1][ids.k[2, 2]] > .9

    mp.pp.k[2] = [.9, .1]
    mp.vp[1][ids.c1[:,2]] = mp.pp.c[2][1][:, 1]
    mp.vp[1][ids.k[:, 2]] = [0.5, 0.5]
    OptimizeElbo.maximize_f(ElboDeriv.elbo, blob, mp, trans, omitted_ids=omitted_ids)
    @test mp.vp[1][ids.k[1, 2]] > .9

    mp.pp.k[2] = [.1, .9]
    mp.vp[1][ids.c1[:,2]] = mp.pp.c[2][1][:, 2]
    mp.vp[1][ids.k[:, 2]] = [0.5, 0.5]
    OptimizeElbo.maximize_f(ElboDeriv.elbo, blob, mp, trans, omitted_ids=omitted_ids)
    @test mp.vp[1][ids.k[2, 2]] > .9
end


function test_bad_a_init()
    gal_color_mode = [ 2.47122, 1.832, 4.0, 5.9192, 9.12822]
    ce = CatalogEntry([7.2, 8.3], false, gal_color_mode, gal_color_mode,
            0.5, .7, pi/4, .5)

    blob0 = SDSS.load_stamp_blob(dat_dir, "164.4311-39.0359")
    for b in 1:5
        blob0[b].H, blob0[b].W = 20, 23
        blob0[b].wcs = WCS.wcs_id
    end
    blob = Synthetic.gen_blob(blob0, [ce,])

    mp = ModelInit.cat_init([ce,])
    trans = get_mp_transform(mp, loc_width=1.0);

    mp.vp[1][ids.a] = [ 0.5, 0.5 ]

    omitted_ids = [ids_free.a]
    OptimizeElbo.maximize_f(ElboDeriv.elbo, blob, mp, trans, omitted_ids=omitted_ids)

    mp.vp[1][ids.a] = [ 0.8, 0.2 ]
    elbo_bad = ElboDeriv.elbo_likelihood(blob, mp)
    @test elbo_bad.d[ids.a[2], 1] > 0

    omitted_ids = setdiff(1:length(UnconstrainedParams), ids_free.a)
    OptimizeElbo.maximize_f(ElboDeriv.elbo, blob, mp, trans, omitted_ids=omitted_ids)
    @test mp.vp[1][ids.a[2]] >= 0.5

    mp2 = deepcopy(mp)
    mp2.vp[1][ids.a] = [ 0.01, 0.99 ]
    elbo_true2 = ElboDeriv.elbo_likelihood(blob, mp2)
    mp2.vp[1][ids.a] = [ 0.99, 0.01 ]
    elbo_bad2 = ElboDeriv.elbo_likelihood(blob, mp2)
    @test elbo_true2.v > elbo_bad2.v
    @test elbo_bad2.d[ids.a[2], 1] > 0
end


function test_likelihood_invariance_to_a()
    fluxes = [2.47122, 1.832, 4.0, 5.9192, 9.12822]
    ce = CatalogEntry([7.2,8.3], false, fluxes, fluxes, 0.5, .7, pi/4, .5)

    blob0 = SDSS.load_stamp_blob(dat_dir, "164.4311-39.0359")
    for b in 1:5
        blob0[b].H, blob0[b].W = 20, 23
        blob0[b].wcs = WCS.wcs_id
    end
    blob = Synthetic.gen_blob(blob0, [ce,])

    mp = ModelInit.cat_init([ce,])
    trans = get_mp_transform(mp, loc_width=1.0);

    mp.vp[1][ids.a] = [ 0.8, 0.2 ]
    omitted_ids = [ids_free.a, ids_free.r2[:]]
    OptimizeElbo.maximize_f(ElboDeriv.elbo_likelihood, blob, mp,
        trans, omitted_ids=omitted_ids)

    mp2 = ModelInit.cat_init([ce,])
    mp2.vp[1][ids.a] = [ 0.2, 0.8 ]
    OptimizeElbo.maximize_f(ElboDeriv.elbo_likelihood, blob, mp2,
        trans, omitted_ids=omitted_ids)

    mp.vp[1][ids.a] = [ 0.5, 0.5 ]
    mp2.vp[1][ids.a] = [ 0.5, 0.5 ]
    @test_approx_eq_eps(ElboDeriv.elbo_likelihood(blob, mp).v,
        ElboDeriv.elbo_likelihood(blob, mp2).v, 1)

    for i in 2:length(1:length(CanonicalParams)) #skip a
        @test_approx_eq_eps mp.vp[1][i] / mp2.vp[1][i] 1. 0.1
    end
end


function test_kl_invariance_to_a()
    fluxes = [2.47122, 1.832, 4.0, 5.9192, 9.12822]
    ce = CatalogEntry([7.2,8.3], false, fluxes, fluxes, 0.5, .7, pi/4, .5)
    blob0 = SDSS.load_stamp_blob(dat_dir, "164.4311-39.0359")
    for b in 1:5
        blob0[b].H, blob0[b].W = 20, 23
        blob0[b].wcs = WCS.wcs_id
    end
    blob = Synthetic.gen_blob(blob0, [ce,])

    kl_wrapper(blob, mp) = begin
        accum = zero_sensitive_float(CanonicalParams)
        ElboDeriv.subtract_kl!(mp, accum)
        accum
    end

    mp = ModelInit.cat_init([ce,])
    trans = get_mp_transform(mp, loc_width=1.0);
    mp.vp[1][ids.a] = [ 0.2, 0.8 ]
    omitted_ids = [ids_free.a;]
    OptimizeElbo.maximize_f(kl_wrapper, blob, mp,
        trans, omitted_ids=omitted_ids, ftol_abs=1e-9)

    mp2 = ModelInit.cat_init([ce,])
    mp2.vp[1][ids.a] = [ 0.8, 0.2 ]
    OptimizeElbo.maximize_f(kl_wrapper, blob, mp2,
        trans, omitted_ids=omitted_ids, ftol_abs=1e-9)

    mp.vp[1][ids.a] = [ 0.5, 0.5 ]
    mp2.vp[1][ids.a] = [ 0.5, 0.5 ]
    @test_approx_eq_eps kl_wrapper(blob, mp).v kl_wrapper(blob, mp2).v 1e-1

    for i in 2:length(1:length(CanonicalParams)) #skip a
        @test_approx_eq_eps mp.vp[1][i] / mp2.vp[1][i] 1. 0.1
    end
end


function test_elbo_invariance_to_a()
    fluxes = [2.47122, 1.832, 4.0, 5.9192, 9.12822] * 100
    ce = CatalogEntry([7.2,8.3], false, fluxes, fluxes, 0.5, .7, pi/4, .5)
    blob0 = SDSS.load_stamp_blob(dat_dir, "164.4311-39.0359")
    for b in 1:5
        blob0[b].H, blob0[b].W = 20, 23
        blob0[b].wcs = WCS.wcs_id
    end
    blob = Synthetic.gen_blob(blob0, [ce,])

    mp = ModelInit.cat_init([ce,])
    trans = get_mp_transform(mp, loc_width=1.0);

    mp.vp[1][ids.a] = [ 0.8, 0.2 ]
    omitted_ids = [ids_free.a, ids_free.r2[:], ids_free.c2[:], ids_free.e_dev]
    OptimizeElbo.maximize_f(ElboDeriv.elbo, blob, mp,
        trans, omitted_ids=omitted_ids)

    mp2 = ModelInit.cat_init([ce,])
    mp2.vp[1][ids.a] = [ 0.2, 0.8 ]
    OptimizeElbo.maximize_f(ElboDeriv.elbo, blob, mp2,
        trans, omitted_ids=omitted_ids)

    mp.vp[1][ids.a] = [ 0.5, 0.5 ]
    mp2.vp[1][ids.a] = [ 0.5, 0.5 ]
    @test_approx_eq_eps ElboDeriv.elbo(blob, mp).v ElboDeriv.elbo(blob, mp2).v 1

    for i in setdiff(1:length(CanonicalParams), ids.a) #skip a
        @test_approx_eq_eps mp.vp[1][i] / mp2.vp[1][i] 1. 0.1
    end
end


function test_peak_init_galaxy_optimization()
    blob, mp, body = gen_sample_galaxy_dataset()
    mp = ModelInit.peak_init(blob)
    trans = get_mp_transform(mp, loc_width=1.0);

    OptimizeElbo.maximize_likelihood(blob, mp, trans)
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
    trans = get_mp_transform(mp, loc_width=1.0);

    @test mp.S == 2

    OptimizeElbo.maximize_likelihood(blob, mp, trans)

    verify_sample_star(mp.vp[1], [11.1, 21.2])
    verify_sample_galaxy(mp.vp[2], [15.3, 31.4])
end


function test_full_elbo_optimization()
    blob, mp, body = gen_sample_galaxy_dataset(perturb=true)
    trans = get_mp_transform(mp, loc_width=1.0);
    OptimizeElbo.maximize_elbo(blob, mp, trans, xtol_rel=0.0)
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

    mp = ModelInit.cat_init(cat_entries)
    trans = get_mp_transform(mp, loc_width=1.0);
    OptimizeElbo.maximize_elbo(blob, mp, trans, xtol_rel=0.0)
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

    mp_good_init = ModelInit.cat_init(cat_coadd)
    trans = get_mp_transform(mp_good_init, loc_width=1.0);
    OptimizeElbo.maximize_elbo(blob, mp_good_init, trans)
    @test mp_good_init.vp[1][ids.a[2]] > .5

    mp_bad_init = ModelInit.cat_init(cat_primary)
    OptimizeElbo.maximize_f(ElboDeriv.elbo, blob, mp_bad_init, trans)
    @test mp_bad_init.vp[1][ids.a[2]] > .5

    @test_approx_eq_eps mp_good_init.vp[1][ids.e_scale] mp_bad_init.vp[1][ids.e_scale] 0.2
    @test_approx_eq_eps mp_good_init.vp[1][ids.e_axis] mp_bad_init.vp[1][ids.e_axis] 0.2
    @test_approx_eq_eps mp_good_init.vp[1][ids.e_dev] mp_bad_init.vp[1][ids.e_dev] 0.2
    @test_approx_eq_eps mp_good_init.vp[1][ids.e_angle] mp_bad_init.vp[1][ids.e_angle] 0.2
end


function test_color(trans::DataTransform)
    blob, mp, body = gen_sample_galaxy_dataset(perturb=true)
    trans = get_mp_transform(mp, loc_width=1.0);

    # these are a bright star's colors
    mp.vp[1][ids.c1[:, 1]] = [2.42824, 1.13996, 0.475603, 0.283062]
    mp.vp[1][ids.c1[:, 2]] = [2.42824, 1.13996, 0.475603, 0.283062]

    klc_wrapper(blob, mp) = begin
        accum = zero_sensitive_float(CanonicalParams, mp.S)
        for s in 1:mp.S, i in 1:2, d in 1:D
            ElboDeriv.subtract_kl_c!(d, i, s, mp, accum)
        end
        accum
    end
    omitted_ids = [ids_free.c1[:]]
    OptimizeElbo.maximize_f(klc_wrapper, blob, mp, trans,
        omitted_ids=omitted_ids, ftol_abs=1e-9)

    @test_approx_eq_eps mp.vp[1][ids.k[2, 1]] 1 1e-2

    @test_approx_eq mp.vp[1][ids.a[2]] 0.01
end


function test_quadratic_optimization(trans::DataTransform)

    # A very simple quadratic function to test the optimization.
    const centers = collect(linrange(0.1, 0.9, length(CanonicalParams)))

    # Set feasible centers for the indicators.
    centers[ids.a] = [ 0.4, 0.6 ]
    centers[ids.k] = [ 0.3 0.3; 0.7 0.7 ]

    function quadratic_function(unused_blob::Blob, mp::ModelParams)
        val = zero_sensitive_float(CanonicalParams)
        val.v = -sum((mp.vp[1] - centers) .^ 2)
        val.d[:] = -2.0 * (mp.vp[1] - centers)

        val
    end

    ###############
    # TODO: this needs to be changed for the new transforms.

    # 0.5 is an innocuous value for all parameters.
    mp = empty_model_params(1)
    trans = DataTransform
    n = length(CanonicalParams)
    mp.vp = convert(VariationalParams{Float64}, [fill(0.5, n) for s in 1:1])
    unused_blob = gen_sample_star_dataset()[1]

    vp_lbs = convert(VariationalParams{Float64}, [fill(1e-6, n) for s in 1:1])
    vp_ubs = convert(VariationalParams{Float64}, [fill(1.0 - 1e-6, n) for s in 1:1])

    lbs = trans.from_vp(vp_lbs)[1]
    ubs = trans.from_vp(vp_ubs)[1]

    OptimizeElbo.maximize_f(quadratic_function, unused_blob, mp, trans, lbs, ubs,
        xtol_rel=1e-16, ftol_abs=1e-16)

    @test_approx_eq_eps mp.vp[1] centers 1e-6
    @test_approx_eq_eps quadratic_function(unused_blob, mp).v 0.0 1e-15
end


####################################################

# test_quadratic_optimization(pixel_rect_transform)
# test_quadratic_optimization(world_rect_transform)
# test_quadratic_optimization(free_transform)

test_objective_wrapper()

#test_bad_galaxy_init()
test_kappa_finding()
test_bad_a_init()
#test_elbo_invariance_to_a()
#test_kl_invariance_to_a()
#test_likelihood_invariance_to_a()
test_star_optimization()

test_full_elbo_optimization(free_transform)
test_galaxy_optimization(free_transform)
test_real_stamp_optimization(free_transform)
