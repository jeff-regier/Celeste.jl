using Celeste
using CelesteTypes
using Base.Test
using SampleData
using Transform
using Compat
import SloanDigitalSkySurvey.WCSUtils

import OptimizeElbo


println("Running optimization tests.")

function verify_sample_star(vs, pos)
    @test vs[ids.a[2]] <= 0.01

    @test_approx_eq_eps vs[ids.u[1]] pos[1] 0.1
    @test_approx_eq_eps vs[ids.u[2]] pos[2] 0.1

    brightness_hat = exp(vs[ids.r1[1]] + 0.5 * vs[ids.r2[1]])
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

    brightness_hat = exp(vs[ids.r1[2]] + 0.5 * vs[ids.r2[2]])
    @test_approx_eq_eps brightness_hat / sample_galaxy_fluxes[3] 1. 0.01

    true_colors = log(sample_galaxy_fluxes[2:5] ./ sample_galaxy_fluxes[1:4])
    for b in 1:4
        @test_approx_eq_eps vs[ids.c1[b, 2]] true_colors[b] 0.2
    end
end


#########################################################

function test_objective_wrapper()
    omitted_ids = Int64[];
    kept_ids = setdiff(1:length(ids_free), omitted_ids);

    blob, mp, bodies, tiled_blob = SampleData.gen_three_body_dataset();
    # Change the tile size.
    tiled_blob, mp = ModelInit.initialize_celeste(
      blob, bodies, tile_width=5, fit_psf=false, patch_radius=10.);
    mp.active_sources = Int64[2, 3]
    trans = Transform.get_mp_transform(mp, loc_width=1.0);

    wrapper =
      OptimizeElbo.ObjectiveWrapperFunctions(
        mp -> ElboDeriv.elbo(tiled_blob, mp),
        mp, trans, kept_ids, omitted_ids);

    x = trans.vp_to_array(mp.vp, omitted_ids);
    elbo_result =
      trans.transform_sensitive_float(ElboDeriv.elbo(tiled_blob, mp), mp);
    elbo_grad = reduce(vcat, [ elbo_result.d[kept_ids, si] for
                               si in 1:length(mp.active_sources) ]);

    # Tese the print function
    wrapper.state.verbose = true
    wrapper.f_objective(x[:]);
    wrapper.state.verbose = false

    w_v, w_grad = wrapper.f_value_grad(x[:]);
    w_grad2 = zeros(Float64, length(x));
    wrapper.f_value_grad!(x[:], w_grad2);

    @test_approx_eq(w_v, elbo_result.v)
    @test_approx_eq(w_grad, elbo_grad)
    @test_approx_eq(w_grad, w_grad2)

    @test_approx_eq(w_v, wrapper.f_value(x[:]))
    @test_approx_eq(w_grad, wrapper.f_grad(x[:]))

    this_iter = wrapper.state.f_evals;
    wrapper.f_value(x[:] + 1.0);
    @test wrapper.state.f_evals == this_iter + 1

    # # Check the AD gradient.
    # ad_grad = wrapper.f_ad_grad(x[:]);
    # @test_approx_eq ad_grad[:] w_grad
end

# function test_objective_hessians()
#     blob, mp, bodies, tiled_blob = SampleData.gen_three_body_dataset();
#     # Change the tile size.
#     tiled_blob, mp = ModelInit.initialize_celeste(
#       blob, bodies, tile_width=5, fit_psf=false, patch_radius=10.);
#     mp.active_sources = Int64[2, 3]
#     trans = Transform.get_mp_transform(mp, loc_width=1.0);
#     omitted_ids = Int64[];
#     kept_ids = setdiff(1:length(ids_free), omitted_ids);
#     x = trans.vp_to_array(mp.vp, omitted_ids);
#
#     wrapper =
#       OptimizeElbo.ObjectiveWrapperFunctions(
#         mp -> ElboDeriv.elbo(tiled_blob, mp),
#         mp, trans, kept_ids, omitted_ids);
#
#     # Test that the Hessian works in its various flavors.
#     println("Testing autodiff Hessian...")
#     w_hess = zeros(Float64, length(x), length(x));
#     wrapper.f_ad_hessian!(x[:], w_hess);
#
#     hess_i, hess_j, hess_val = wrapper.f_ad_hessian_sparse(x[:]);
#     w_hess_sparse =
#       OptimizeElbo.unpack_hessian_vals(hess_i, hess_j, hess_val, size(x));
#     @test_approx_eq(w_hess, full(w_hess_sparse))
#
#     println("Testing slow autodiff Hessian...")
#     wrapper_slow_hess =
#       OptimizeElbo.ObjectiveWrapperFunctions(
#         mp -> ElboDeriv.elbo(tiled_blob, mp),
#         mp, trans, kept_ids, omitted_ids, fast_hessian=false);
#
#     slow_w_hess = zeros(Float64, length(x), length(x));
#     wrapper_slow_hess.f_ad_hessian!(x[:], slow_w_hess);
#     @test_approx_eq(slow_w_hess, w_hess)
#
#     hess_i, hess_j, hess_val = wrapper_slow_hess.f_ad_hessian_sparse(x[:]);
#     slow_w_hess_sparse =
#       OptimizeElbo.unpack_hessian_vals(hess_i, hess_j, hess_val, size(x));
#     @test_approx_eq(slow_w_hess, full(slow_w_hess_sparse))
# end


function test_star_optimization()
    blob, mp, body, tiled_blob = gen_sample_star_dataset();

    # Newton's method converges on a small galaxy unless we start with
    # a high star probability.
    mp.vp[1][ids.a] = [0.8, 0.2]
    transform = get_mp_transform(mp, loc_width=1.0);
    OptimizeElbo.maximize_likelihood(tiled_blob, mp, transform, verbose=false)
    verify_sample_star(mp.vp[1], [10.1, 12.2])
end


function test_single_source_optimization()
  blob, mp, three_bodies, tiled_blob = gen_three_body_dataset();

  # Change the tile size.
  tiled_blob, mp = ModelInit.initialize_celeste(
  blob, three_bodies, tile_width=10, fit_psf=false);
  mp_original = deepcopy(mp);

  s = 2
  mp.active_sources = Int64[s]
  transform = get_mp_transform(mp, loc_width=1.0);

  f = ElboDeriv.elbo;
  omitted_ids = Int64[]

  ElboDeriv.elbo_likelihood(tiled_blob, mp).v

  OptimizeElbo.maximize_likelihood(tiled_blob, mp, transform, verbose=true)

  # Test that it only optimized source s
  @test mp.vp[s] != mp_original.vp[s]
  for other_s in setdiff(1:mp.S, s)
    @test_approx_eq mp.vp[other_s] mp_original.vp[other_s]
  end
end


function test_two_body_optimization_newton()
    # This test is currently too slow to be part of the ordinary
    # test suite, and the block diagonal hessian does not work very well.
    # For now, leave it in for future reference.

    blob, mp, two_bodies, tiled_blob = SampleData.gen_two_body_dataset();

    trans = get_mp_transform(mp, loc_width=1.0);
    function lik_function(tiled_blob::TiledBlob, mp::ModelParams)
      ElboDeriv.elbo_likelihood(tiled_blob, mp)
    end
    omitted_ids = [ids_free.k[:]; ids_free.c2[:]; ids_free.r2]

    function elbo_function(tiled_blob::TiledBlob, mp::ModelParams)
      ElboDeriv.elbo(tiled_blob, mp)
    end
    omitted_ids = Int64[]

    mp_newton = deepcopy(mp);
    newton_iter_count = OptimizeElbo.maximize_f_newton(
      elbo_function, tiled_blob, mp_newton, trans,
      omitted_ids=omitted_ids, verbose=true);

    mp_bfgs = deepcopy(mp);
    bfgs_iter_count = OptimizeElbo.maximize_f(
      elbo_function, tiled_blob, mp_bfgs, trans,
      omitted_ids=omitted_ids, verbose=true);

    newton_image =
      ElboDeriv.tile_predicted_image(tiled_blob[3][1,1], mp_newton,
                                     mp_newton.tile_sources[3][1,1]);
    bfgs_image =
      ElboDeriv.tile_predicted_image(tiled_blob[3][1,1], mp_bfgs,
                                     mp_bfgs.tile_sources[3][1,1]);
    original_image = tiled_blob[3][1,1].pixels;

    PyPlot.figure()
    PyPlot.subplot(1, 3, 1)
    PyPlot.imshow(newton_image)
    PyPlot.title("Newton")

    PyPlot.subplot(1, 3, 2)
    PyPlot.imshow(bfgs_image)
    PyPlot.title("BFGS")

    PyPlot.subplot(1, 3, 3)
    PyPlot.imshow(original_image)
    PyPlot.title("Original")

    sum((newton_image .- original_image) .^ 2)
    sum((bfgs_image .- original_image) .^ 2)

    # newton beats bfgs on the elbo, though not on the likelihood.
    elbo_function(tiled_blob, mp_bfgs).v
    elbo_function(tiled_blob, mp_newton).v
end


function test_galaxy_optimization()
    # NLOpt fails here so use newton.
    blob, mp, body, tiled_blob = gen_sample_galaxy_dataset();
    trans = get_mp_transform(mp, loc_width=3.0);
    OptimizeElbo.maximize_likelihood(tiled_blob, mp, trans, verbose=false)
    verify_sample_galaxy(mp.vp[1], [8.5, 9.6])
end


function test_kappa_finding()
    blob, mp, body, tiled_blob = gen_sample_galaxy_dataset();
    trans = get_mp_transform(mp, loc_width=1.0);
    omitted_ids = setdiff(1:length(UnconstrainedParams), ids_free.k[:])

    get_kl_gal_c() = begin
        accum = zero_sensitive_float(CanonicalParams)
        for d in 1:D
            ElboDeriv.subtract_kl_c(d, 2, 1, mp, accum)
        end
        -accum.v
    end

    mp.vp[1][ids.k[:, 2]] = [0.01, 0.99]
    mp.vp[1][ids.c1[:,2]] = mp.pp.c_mean[:, 2, 2]
    lower_klc = get_kl_gal_c()
    mp.vp[1][ids.c1[:,2]] = mp.pp.c_mean[:, 1, 2]
    higher_klc = get_kl_gal_c()
    @test lower_klc < higher_klc

    mp.vp[1][ids.k[:, 2]] = [0.99, 0.01]
    mp.vp[1][ids.c1[:,2]] = mp.pp.c_mean[:, 1, 2]
    lower_klc = get_kl_gal_c()
    mp.vp[1][ids.c1[:,2]] = mp.pp.c_mean[:, 2, 2]
    higher_klc = get_kl_gal_c()
    @test lower_klc < higher_klc

    mp.pp.c_cov[:, :, 1, 2] = mp.pp.c_cov[:, :, 2, 2] = eye(4)
    klc_wrapper{NumType <: Number}(
        tiled_blob::TiledBlob, mp::ModelParams{NumType}) = begin
      accum = zero_sensitive_float(CanonicalParams, NumType)
      for d in 1:D
          ElboDeriv.subtract_kl_c(d, 2, 1, mp, accum)
      end
      accum
    end

    mp.vp[1][ids.c1[:,2]] = mp.pp.c_mean[:, 1, 2]
    mp.vp[1][ids.k[:, 2]] = [0.5, 0.5]
    lbs, ubs = OptimizeElbo.get_nlopt_unconstrained_bounds(
      mp.vp, omitted_ids, trans);
    OptimizeElbo.maximize_f_bfgs(
      klc_wrapper, tiled_blob, mp, trans, lbs, ubs, omitted_ids=omitted_ids)
    @test mp.vp[1][ids.k[1, 2]] > .9

    mp.vp[1][ids.c1[:,2]] = mp.pp.c_mean[:, 2, 2]
    mp.vp[1][ids.k[:, 2]] = [0.5, 0.5]
    OptimizeElbo.maximize_f_bfgs(
      klc_wrapper, tiled_blob, mp, trans, lbs, ubs, omitted_ids=omitted_ids)
    @test mp.vp[1][ids.k[2, 2]] > .9

    mp.pp.k[:, 2] = [.9, .1]
    mp.vp[1][ids.c1[:,2]] = mp.pp.c_mean[:, 1, 2]
    mp.vp[1][ids.k[:, 2]] = [0.5, 0.5]
    OptimizeElbo.maximize_f_bfgs(
      ElboDeriv.elbo, tiled_blob, mp, trans, lbs, ubs, omitted_ids=omitted_ids)
    @test mp.vp[1][ids.k[1, 2]] > .9

    mp.pp.k[:, 2] = [.1, .9]
    mp.vp[1][ids.c1[:,2]] = mp.pp.c_mean[:, 2, 2]
    mp.vp[1][ids.k[:, 2]] = [0.5, 0.5]
    OptimizeElbo.maximize_f_bfgs(
      ElboDeriv.elbo, tiled_blob, mp, trans, lbs, ubs, omitted_ids=omitted_ids)
    @test mp.vp[1][ids.k[2, 2]] > .9
end


function test_bad_a_init()
    gal_color_mode = [ 2.47122, 1.832, 4.0, 5.9192, 9.12822]
    ce = CatalogEntry([7.2, 8.3], false, gal_color_mode, gal_color_mode,
            0.5, .7, pi/4, .5, "test")

    blob0 = SkyImages.load_stamp_blob(dat_dir, "164.4311-39.0359");
    for b in 1:5
        blob0[b].H, blob0[b].W = 20, 23
        blob0[b].wcs = WCSUtils.wcs_id
    end
    blob = Synthetic.gen_blob(blob0, [ce,])

    tiled_blob, mp = ModelInit.initialize_celeste(blob, [ce,])
    trans = get_mp_transform(mp, loc_width=1.0);

    mp.vp[1][ids.a] = [ 0.5, 0.5 ]

    # Use BFGS because Newton doesn't work well with non-convex problems.
    omitted_ids = [ids_free.a]
    lbs, ubs = OptimizeElbo.get_nlopt_unconstrained_bounds(
      mp.vp, omitted_ids, trans);
    OptimizeElbo.maximize_f_bfgs(
      ElboDeriv.elbo, tiled_blob, mp, trans, lbs, ubs, omitted_ids=omitted_ids)

    mp.vp[1][ids.a] = [ 0.8, 0.2 ]
    elbo_bad = ElboDeriv.elbo_likelihood(tiled_blob, mp)
    @test elbo_bad.d[ids.a[2], 1] > 0

    omitted_ids = setdiff(1:length(UnconstrainedParams), ids_free.a)
    lbs, ubs = OptimizeElbo.get_nlopt_unconstrained_bounds(
      mp.vp, omitted_ids, trans);
    OptimizeElbo.maximize_f_bfgs(
      ElboDeriv.elbo, tiled_blob, mp, trans, lbs, ubs, omitted_ids=omitted_ids)
    @test mp.vp[1][ids.a[2]] >= 0.5

    mp2 = deepcopy(mp)
    mp2.vp[1][ids.a] = [ 0.01, 0.99 ]
    elbo_true2 = ElboDeriv.elbo_likelihood(tiled_blob, mp2)
    mp2.vp[1][ids.a] = [ 0.99, 0.01 ]
    elbo_bad2 = ElboDeriv.elbo_likelihood(tiled_blob, mp2)
    @test elbo_true2.v > elbo_bad2.v
    @test elbo_bad2.d[ids.a[2], 1] > 0
end


function test_full_elbo_optimization()
    blob, mp, body, tiled_blob = gen_sample_galaxy_dataset(perturb=true);
    trans = get_mp_transform(mp, loc_width=1.0);
    OptimizeElbo.maximize_elbo(tiled_blob, mp, trans, xtol_rel=0.0);
    verify_sample_galaxy(mp.vp[1], [8.5, 9.6]);
end


function test_real_stamp_optimization()
    blob = SkyImages.load_stamp_blob(dat_dir, "5.0073-0.0739");
    cat_entries = SkyImages.load_stamp_catalog(dat_dir, "s82-5.0073-0.0739", blob);
    bright(ce) = sum(ce.star_fluxes) > 3 || sum(ce.gal_fluxes) > 3
    cat_entries = filter(bright, cat_entries);
    inbounds(ce) = ce.pos[1] > -10. && ce.pos[2] > -10 &&
        ce.pos[1] < 61 && ce.pos[2] < 61
    cat_entries = filter(inbounds, cat_entries);

    tiled_blob, mp = ModelInit.initialize_celeste(blob, cat_entries);
    trans = get_mp_transform(mp, loc_width=1.0);
    OptimizeElbo.maximize_elbo(tiled_blob, mp, trans, xtol_rel=0.0);
end


function test_color()
    # TODO: Why was this commented out?  Why is it not passing?

    blob, mp, body, tiled_blob = gen_sample_galaxy_dataset(perturb=true);
    trans = get_mp_transform(mp, loc_width=1.0);

    # these are a bright star's colors
    mp.vp[1][ids.c1[:, 1]] = [2.42824, 1.13996, 0.475603, 0.283062]
    mp.vp[1][ids.c1[:, 2]] = [2.42824, 1.13996, 0.475603, 0.283062]

    klc_wrapper{NumType <: Number}(
        tiled_blob::TiledBlob, mp::ModelParams{NumType}) = begin
      accum = zero_sensitive_float(CanonicalParams, NumType, mp.S)
      for s in 1:mp.S, i in 1:2, d in 1:D
          ElboDeriv.subtract_kl_c(d, i, s, mp, accum)
      end
      accum
    end
    omitted_ids = [ids_free.c1[:]]
    OptimizeElbo.maximize_f(klc_wrapper, tiled_blob, mp, trans,
        omitted_ids=omitted_ids, ftol_abs=1e-9)

    @test mp.vp[1][ids.a[2]] <= 0.01
    @test_approx_eq_eps mp.vp[1][ids.k[2, 1]] 1 1e-2
end


function test_quadratic_optimization()
    println("Testing quadratic optimization.")

    # A very simple quadratic function to test the optimization.
    const centers = collect(linspace(0.1, 0.9, length(CanonicalParams)));

    # Set feasible centers for the indicators.
    centers[ids.a] = [ 0.4, 0.6 ]
    centers[ids.k] = [ 0.3 0.3; 0.7 0.7 ]

    function quadratic_function{NumType <: Number}(
          unused_blob::TiledBlob, mp::ModelParams{NumType})

        val = zero_sensitive_float(CanonicalParams, NumType)
        val.v = -sum((mp.vp[1] - centers) .^ 2)
        val.d[:] = -2.0 * (mp.vp[1] - centers)
        val.h[:, :] = diagm(fill(-2.0, length(CanonicalParams)))
        val
    end

    bounds = Array(ParamBounds, 1)
    bounds[1] = ParamBounds()
    for param in setdiff(fieldnames(ids), [:a, :k])
      bounds[1][symbol(param)] = fill(ParamBox(0., 1.0, 1.0), length(ids.(param)))
    end
    bounds[1][:a] = [ SimplexBox(0.0, 1.0, 2) ]
    bounds[1][:k] = fill(SimplexBox(0.0, 1.0, 2), 2)
    trans = DataTransform(bounds);

    mp = empty_model_params(1);
    n = length(CanonicalParams)
    mp.vp = convert(VariationalParams{Float64}, [fill(0.5, n) for s in 1:1]);
    unused_blob = gen_sample_star_dataset()[4];

    OptimizeElbo.maximize_f(
        quadratic_function, unused_blob, mp, trans,
        xtol_rel=1e-16, ftol_abs=1e-16)

    @test_approx_eq_eps mp.vp[1] centers 1e-6
    @test_approx_eq_eps quadratic_function(unused_blob, mp).v 0.0 1e-15
end

####################################################

test_quadratic_optimization()
test_objective_wrapper()
#test_objective_hessians()
test_star_optimization()
test_galaxy_optimization()
test_single_source_optimization()
test_full_elbo_optimization()
test_real_stamp_optimization()

# These tests are commented out because they mainly test NLopt,
# which we are no longer using. It isn't straightforward to convert
# them to testing trust region optimization because they are 1D
# optimization problems.
#test_kappa_finding()
#test_bad_a_init()
