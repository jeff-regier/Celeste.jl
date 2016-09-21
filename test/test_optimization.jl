using Base.Test

using Celeste: Model, Transform, SensitiveFloats


function verify_sample_star(vs, pos)
    @test vs[ids.a[2, 1]] <= 0.01

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
    @test vs[ids.a[2, 1]] >= 0.99

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
    omitted_ids = Int[];
    kept_ids = setdiff(1:length(ids_free), omitted_ids);

    blob, ea, bodies = SampleData.gen_three_body_dataset();
    # Change the tile size.
    ea = make_elbo_args(
      blob, bodies, tile_width=5, fit_psf=false, patch_radius_pix=10.);
    ea.active_sources = Int[2, 3]
    trans = Transform.get_mp_transform(ea.vp, ea.active_sources, loc_width=1.0);

    wrapper =
      DeterministicVI.ObjectiveWrapperFunctions(
        ea -> DeterministicVI.elbo(ea),
        ea, trans, kept_ids, omitted_ids);

    x = trans.vp_to_array(ea.vp, omitted_ids);
    elbo_result =
      trans.transform_sensitive_float(DeterministicVI.elbo(ea), ea.vp, ea.active_sources);
    elbo_grad = reduce(vcat, [ elbo_result.d[kept_ids, si] for
                               si in 1:length(ea.active_sources) ]);

    # Tese the print function
    wrapper.state.verbose = true
    wrapper.f_objective(x[:]);
    wrapper.state.verbose = false

    w_v, w_grad = wrapper.f_value_grad(x[:]);
    w_grad2 = zeros(Float64, length(x));
    wrapper.f_value_grad!(x[:], w_grad2);

    @test_approx_eq(w_v, elbo_result.v[1])
    @test_approx_eq(w_grad, elbo_grad)
    @test_approx_eq(w_grad, w_grad2)

    @test_approx_eq(w_v, wrapper.f_value(x[:]))
    @test_approx_eq(w_grad, wrapper.f_grad(x[:]))

    this_iter = wrapper.state.f_evals;
    wrapper.f_value(x[:] + 1.0);
    @test wrapper.state.f_evals == this_iter + 1
end


function test_star_optimization()
    blob, ea, body = gen_sample_star_dataset();

    # Newton's method converges on a small galaxy unless we start with
    # a high star probability.
    ea.vp[1][ids.a[:, 1]] = [0.8, 0.2]
    DeterministicVI.maximize_f(DeterministicVI.elbo_likelihood, ea; loc_width=1.0)
    verify_sample_star(ea.vp[1], [10.1, 12.2])
end


function test_single_source_optimization()
    blob, ea, three_bodies = gen_three_body_dataset();

    # Change the tile size.
    ea = make_elbo_args(blob, three_bodies, tile_width=10, fit_psf=false);
    ea_original = deepcopy(ea);

    s = 2
    ea.active_sources = Int[s]
    omitted_ids = Int[]
    DeterministicVI.elbo_likelihood(ea).v[1]
    DeterministicVI.maximize_f(DeterministicVI.elbo_likelihood, ea; loc_width=1.0)

    # Test that it only optimized source s
    @test ea.vp[s] != ea_original.vp[s]
    for other_s in setdiff(1:ea.S, s)
        @test_approx_eq ea.vp[other_s] ea_original.vp[other_s]
    end
end


function test_two_body_optimization_newton()
    # This test is currently too slow to be part of the ordinary
    # test suite, and the block diagonal hessian does not work very well.
    # For now, leave it in for future reference.

    blob, ea, two_bodies, tiled_blob = SampleData.gen_two_body_dataset();

    trans = get_mp_transform(ea.vp, ea.active_sources, loc_width=1.0);
    function lik_function(tiled_blob::Vector{TiledImage}, ea::ElboArgs)
      DeterministicVI.elbo_likelihood(tiled_blob, ea)
    end
    omitted_ids = [ids_free.k[:]; ids_free.c2[:]; ids_free.r2]

    function elbo_function(tiled_blob::Vector{TiledImage}, ea::ElboArgs)
      DeterministicVI.elbo(tiled_blob, ea)
    end
    omitted_ids = Int[]

    ea_newton = deepcopy(ea);
    newton_iter_count = DeterministicVI.maximize_f_newton(
      elbo_function, tiled_blob, ea_newton, trans,
      omitted_ids=omitted_ids, verbose=true);

    ea_bfgs = deepcopy(ea);
    bfgs_iter_count = DeterministicVI.maximize_f(
      elbo_function, tiled_blob, ea_bfgs, trans,
      omitted_ids=omitted_ids, verbose=true);

    newton_image =
      DeterministicVI.tile_predicted_image(tiled_blob[3][1,1], ea_newton,
                                     ea_newton.tile_source_map[3][1,1]);
    bfgs_image =
      DeterministicVI.tile_predicted_image(tiled_blob[3][1,1], ea_bfgs,
                                     ea_bfgs.tile_source_map[3][1,1]);
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
    elbo_function(tiled_blob, ea_bfgs).v[1]
    elbo_function(tiled_blob, ea_newton).v[1]
end


function test_galaxy_optimization()
    # NLOpt fails here so use newton.
    blob, ea, body = gen_sample_galaxy_dataset();
    DeterministicVI.maximize_f(DeterministicVI.elbo_likelihood, ea; loc_width=3.0)
    verify_sample_galaxy(ea.vp[1], [8.5, 9.6])
end


function test_full_elbo_optimization()
    blob, ea, body = gen_sample_galaxy_dataset(perturb=true);
    DeterministicVI.maximize_f(DeterministicVI.elbo, ea; loc_width=1.0, xtol_rel=0.0);
    verify_sample_galaxy(ea.vp[1], [8.5, 9.6]);
end


function test_real_stamp_optimization()
    blob = SampleData.load_stamp_blob(datadir, "5.0073-0.0739_2kpsf");
    cat_entries = SampleData.load_stamp_catalog(datadir, "s82-5.0073-0.0739_2kpsf", blob);
    bright(ce) = sum(ce.star_fluxes) > 3 || sum(ce.gal_fluxes) > 3
    cat_entries = filter(bright, cat_entries);
    inbounds(ce) = ce.pos[1] > -10. && ce.pos[2] > -10 &&
        ce.pos[1] < 61 && ce.pos[2] < 61
    cat_entries = filter(inbounds, cat_entries);

    ea = make_elbo_args(blob, cat_entries);
    DeterministicVI.maximize_f(DeterministicVI.elbo, ea; loc_width=1.0, xtol_rel=0.0);
end


function test_quadratic_optimization()
    println("Testing quadratic optimization.")

    # A very simple quadratic function to test the optimization.
    const centers = collect(linspace(0.1, 0.9, length(CanonicalParams)));

    # Set feasible centers for the indicators.
    centers[ids.a[:, 1]] = [ 0.4, 0.6 ]
    centers[ids.k] = [ 0.3 0.3; 0.7 0.7 ]

    function quadratic_function{NumType <: Number}(ea::ElboArgs{NumType})
        val = zero_sensitive_float(CanonicalParams, NumType)
        val.v[1] = -sum((ea.vp[1] - centers) .^ 2)
        val.d[:] = -2.0 * (ea.vp[1] - centers)
        val.h[:, :] = diagm(fill(-2.0, length(CanonicalParams)))
        val
    end

    bounds = Array(ParamBounds, 1)
    bounds[1] = ParamBounds()
    for param in setdiff(fieldnames(ids), [:a, :k])
      bounds[1][Symbol(param)] = fill(ParamBox(0., 1.0, 1.0), length(getfield(ids, param)))
    end
    bounds[1][:a] = [ SimplexBox(0.0, 1.0, 2) ]
    bounds[1][:k] = fill(SimplexBox(0.0, 1.0, 2), 2)
    trans = DataTransform(bounds);

    ea = empty_model_params(1);
    n = length(CanonicalParams)
    ea.vp = convert(VariationalParams{Float64}, [fill(0.5, n) for s in 1:1]);

    DeterministicVI.maximize_f(quadratic_function, ea, trans;
                            xtol_rel=1e-16, ftol_abs=1e-16)

    @test_approx_eq_eps ea.vp[1] centers 1e-6
    @test_approx_eq_eps quadratic_function(ea).v[1] 0.0 1e-15
end


test_quadratic_optimization()
test_objective_wrapper()
test_star_optimization()
test_galaxy_optimization()
test_single_source_optimization()
test_full_elbo_optimization()
test_real_stamp_optimization()
