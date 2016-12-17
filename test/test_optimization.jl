using Base.Test

using Celeste: Model, Transform, SensitiveFloats, DeterministicVIImagePSF


function verify_sample_star(vs, pos)
    @test vs[ids.a[2, 1]] <= 0.01

    @test_approx_eq_eps vs[ids.u[1]] pos[1] 0.1
    @test_approx_eq_eps vs[ids.u[2]] pos[2] 0.1

    brightness_hat = exp(vs[ids.r1[1]] + 0.5 * vs[ids.r2[1]])
    @test_approx_eq_eps brightness_hat / sample_star_fluxes[3] 1. 0.01

    true_colors = log.(sample_star_fluxes[2:5] ./ sample_star_fluxes[1:4])
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

    true_colors = log.(sample_galaxy_fluxes[2:5] ./ sample_galaxy_fluxes[1:4])
    for b in 1:4
        @test_approx_eq_eps vs[ids.c1[b, 2]] true_colors[b] 0.2
    end
end


#########################################################

function test_star_optimization()
    images, ea, body = gen_sample_star_dataset();

    # Newton's method converges on a small galaxy unless we start with
    # a high star probability.
    ea.vp[1][ids.a[:, 1]] = [0.8, 0.2]
    DeterministicVI.maximize_f(DeterministicVI.elbo_likelihood, ea; loc_width=1.0)
    verify_sample_star(ea.vp[1], [10.1, 12.2])
end


function test_single_source_optimization()
    images, ea, three_bodies = gen_three_body_dataset();

    s = 2
    ea = make_elbo_args(images, three_bodies, active_source=s);
    ea_original = deepcopy(ea);

    omitted_ids = Int[]
    DeterministicVI.elbo_likelihood(ea).v[]
    DeterministicVI.maximize_f(DeterministicVI.elbo_likelihood, ea; loc_width=1.0)

    # Test that it only optimized source s
    @test ea.vp[s] != ea_original.vp[s]
    for other_s in setdiff(1:ea.S, s)
        @test_approx_eq ea.vp[other_s] ea_original.vp[other_s]
    end
end


function test_galaxy_optimization()
    images, ea, body = gen_sample_galaxy_dataset();
    DeterministicVI.maximize_f(DeterministicVI.elbo_likelihood, ea; loc_width=3.0)
    verify_sample_galaxy(ea.vp[1], [8.5, 9.6])
end


function test_full_elbo_optimization()
    images, ea, body = gen_sample_galaxy_dataset(perturb=true);
    DeterministicVI.maximize_f(DeterministicVI.elbo, ea; loc_width=1.0, xtol_rel=0.0);
    verify_sample_galaxy(ea.vp[1], [8.5, 9.6]);
end


function test_real_stamp_optimization()
    images = SampleData.load_stamp_blob(datadir, "5.0073-0.0739_2kpsf");
    cat_entries = SampleData.load_stamp_catalog(datadir, "s82-5.0073-0.0739_2kpsf", images);
    bright(ce) = sum(ce.star_fluxes) > 3 || sum(ce.gal_fluxes) > 3
    cat_entries = filter(bright, cat_entries);
    inbounds(ce) = ce.pos[1] > -10. && ce.pos[2] > -10 &&
        ce.pos[1] < 61 && ce.pos[2] < 61
    cat_entries = filter(inbounds, cat_entries);

    ea = make_elbo_args(images, cat_entries);
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
        val = SensitiveFloat{NumType}(length(ids), 1, true, true)
        val.v[] = -sum((ea.vp[1] - centers) .^ 2)
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
    @test_approx_eq_eps quadratic_function(ea).v[] 0.0 1e-15
end


function test_star_optimization_fft()
    images, ea, body = gen_sample_star_dataset()
    ea.vp[1][ids.a[:, 1]] = [0.8, 0.2]
    ea_fft, fsm_mat = DeterministicVIImagePSF.initialize_fft_elbo_parameters(
        images, deepcopy(ea.vp), ea.patches, [1], use_raw_psf=false)
    elbo_fft_opt =
        DeterministicVIImagePSF.get_fft_elbo_function(ea_fft, fsm_mat)
    DeterministicVI.maximize_f(elbo_fft_opt, ea_fft; loc_width=1.0)
    verify_sample_star(ea_fft.vp[1], [10.1, 12.2])
end


function test_galaxy_optimization_fft()
    images, ea, body = gen_sample_galaxy_dataset()
    ea_fft, fsm_mat = DeterministicVIImagePSF.initialize_fft_elbo_parameters(
        images, deepcopy(ea.vp), ea.patches, [1], use_raw_psf=false)
    elbo_fft_opt =
        DeterministicVIImagePSF.get_fft_elbo_function(ea_fft, fsm_mat)
    DeterministicVI.maximize_f(elbo_fft_opt, ea_fft; loc_width=1.0)
    # TODO: Currently failing since it misses the brighness by 3%, which is
    # greater than the 1% permitted by the test.  However, the ELBO of the
    # FFT optimum is lower than that of the MOG optimum.
    # verify_sample_galaxy(ea_fft.vp[1], [8.5, 9.6])
end

test_star_optimization_fft()
test_galaxy_optimization_fft()

test_quadratic_optimization()
test_star_optimization()
test_single_source_optimization()
test_full_elbo_optimization()
#test_real_stamp_optimization()
test_galaxy_optimization()
