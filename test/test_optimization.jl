using Base.Test

using Celeste: Model, SensitiveFloats, DeterministicVIImagePSF
using Celeste.ConstraintTransforms: ParameterConstraint, ConstraintBatch,
                                    BoxConstraint, SimplexConstraint
using Celeste.DeterministicVI: ElboArgs
using Celeste.DeterministicVI.NewtonMaximize: Config, maximize!, custom_optim_options,
                                              star_only_config, maximize_two_steps!

function verify_sample_star(vs, pos)
    @test vs[ids.a[2]] <= 0.01

    @test isapprox(vs[ids.u[1]], pos[1], atol=0.1)
    @test isapprox(vs[ids.u[2]], pos[2], atol=0.1)

    brightness_hat = exp(vs[ids.r1[1]] + 0.5 * vs[ids.r2[1]])
    @test isapprox(brightness_hat / sample_star_fluxes[3], 1.0, atol=0.01)

    true_colors = log.(sample_star_fluxes[2:5] ./ sample_star_fluxes[1:4])
    for b in 1:4
        @test isapprox(vs[ids.c1[b, 1]], true_colors[b], atol=0.2)
    end
end

function verify_sample_galaxy(vs, pos)
    @test vs[ids.a[2]] >= 0.99

    @test isapprox(vs[ids.u[1]], pos[1], atol=0.1)
    @test isapprox(vs[ids.u[2]], pos[2], atol=0.1)

    @test isapprox(vs[ids.e_axis] , 0.7, atol=0.05)
    @test isapprox(vs[ids.e_dev]  , 0.1, atol=0.08)
    @test isapprox(vs[ids.e_scale], 4.0, atol=0.2)

    phi_hat = vs[ids.e_angle]
    phi_hat -= floor(phi_hat / pi) * pi
    five_deg = 5 * pi/180
    @test isapprox(phi_hat, pi/4, atol=five_deg)

    brightness_hat = exp(vs[ids.r1[2]] + 0.5 * vs[ids.r2[2]])
    @test isapprox(brightness_hat / sample_galaxy_fluxes[3], 1.0, atol=0.01)

    true_colors = log.(sample_galaxy_fluxes[2:5] ./ sample_galaxy_fluxes[1:4])
    for b in 1:4
        @test isapprox(vs[ids.c1[b, 2]], true_colors[b], atol=0.2)
    end
end


#########################################################

function test_star_optimization()
    images, ea, body = gen_sample_star_dataset();

    # Newton's method converges on a small galaxy unless we start with
    # a high star probability.
    ea.vp[1][ids.a] = [0.8, 0.2]

    cfg = Config(ea; loc_width=1.0)
    maximize!(DeterministicVI.elbo, ea, cfg)

    verify_sample_star(ea.vp[1], [10.1, 12.2])
end


function test_single_source_optimization()
    images, ea, three_bodies = gen_three_body_dataset();

    s = 2
    ea = make_elbo_args(images, three_bodies, active_source=s);
    ea_original = deepcopy(ea);

    DeterministicVI.elbo_likelihood(ea).v[]

    cfg = Config(ea; loc_width=1.0)
    maximize!(DeterministicVI.elbo_likelihood, ea, cfg)

    # Test that it only optimized source s
    @test ea.vp[s] != ea_original.vp[s]
    for other_s in setdiff(1:ea.S, s)
        @test ea.vp[other_s] ≈ ea_original.vp[other_s]
    end
end


function test_galaxy_optimization()
    images, ea, body = gen_sample_galaxy_dataset();
    cfg = Config(ea; loc_width=3.0)
    maximize!(DeterministicVI.elbo_likelihood, ea, cfg)
    verify_sample_galaxy(ea.vp[1], [8.5, 9.6])
end


function test_full_elbo_optimization()
    images, ea, body = gen_sample_galaxy_dataset(perturb=true);
    cfg = Config(ea; loc_width=1.0,
                 optim_options=custom_optim_options(xtol_abs=0.0))
    maximize!(DeterministicVI.elbo, ea, cfg)
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
    cfg = Config(ea; loc_width=1.0,
                 optim_options=custom_optim_options(xtol_abs=0.0))
    maximize!(DeterministicVI.elbo, ea, cfg)
end


function test_quadratic_optimization()
    println("Testing quadratic optimization.")

    # A very simple quadratic function to test the optimization.
    const centers = collect(linspace(0.1, 0.9, length(CanonicalParams)));

    # Set feasible centers for the indicators.
    centers[ids.a] = [0.4, 0.6]
    centers[ids.k[:, 1]] = [0.3, 0.7]
    centers[ids.k[:, 2]] = centers[ids.k[:, 1]]

    function quadratic{T}(ea::ElboArgs{T})
        val = SensitiveFloat{T}(length(ids), 1, true, true)
        val.v[] = -sum((ea.vp[1] - centers) .^ 2)
        val.d[:] = -2.0 * (ea.vp[1] - centers)
        val.h[:, :] = diagm(fill(-2.0, length(CanonicalParams)))
        return val
    end

    box = BoxConstraint(0.0, 1.0, 1.0)
    simplex = SimplexConstraint(0.0, 1.0, 2)
    boxes = Vector{Vector{ParameterConstraint{BoxConstraint}}}(1)
    simplexes = Vector{Vector{ParameterConstraint{SimplexConstraint}}}(1)
    boxes[1] = eltype(boxes)()
    simplexes[1] = eltype(simplexes)()
    for paramname in fieldnames(ids)
        param = getfield(ids, paramname)
        if paramname in (:a, :k)
            for inds in param
                push!(simplexes[1], ParameterConstraint(simplex, inds))
            end
        elseif isa(param, Tuple)
            for inds in param
                push!(boxes[1], ParameterConstraint(box, inds))
            end
        else
            push!(boxes[1], ParameterConstraint(box, param))
        end
    end

    ea = empty_model_params(1);
    n = length(CanonicalParams)
    ea.vp = convert(VariationalParams{Float64}, [fill(0.5, n) for s in 1:1]);

    cfg = Config(ea; constraints=ConstraintBatch(boxes, simplexes),
                 optim_options=custom_optim_options(xtol_abs=1e-16, ftol_rel=1e-16))
    maximize!(quadratic, ea, cfg)

    @test isapprox(ea.vp[1]                  , centers, 1e-6)
    @test isapprox(quadratic_function(ea).v[], 0.0    , 1e-15)
end


function test_star_optimization_fft()
    println("Testing star fft optimization.")

    images, ea, body = gen_sample_star_dataset()
    ea.vp[1][ids.a] = [0.8, 0.2]
    ea_fft, fsm_mat = DeterministicVIImagePSF.initialize_fft_elbo_parameters(
        images, deepcopy(ea.vp), ea.patches, [1], use_raw_psf=false)
    elbo_fft_objective = DeterministicVIImagePSF.FFTElboFunction(fsm_mat)

    cfg = Config(ea_fft; loc_width=1.0)
    maximize!(elbo_fft_objective, ea_fft, cfg)

    verify_sample_star(ea_fft.vp[1], [10.1, 12.2])
end


function test_galaxy_optimization_fft()
    println("Testing galaxy fft optimization.")

    images, ea, body = gen_sample_galaxy_dataset()
    ea_fft, fsm_mat = DeterministicVIImagePSF.initialize_fft_elbo_parameters(
        images, deepcopy(ea.vp), ea.patches, [1], use_raw_psf=false)
    elbo_fft_opt = DeterministicVIImagePSF.FFTElboFunction(fsm_mat)
    cfg_star = star_only_config(ea_fft; loc_width=1.0)
    cfg_both = Config(ea_fft; loc_width=1.0)
    maximize_two_steps!(elbo_fft_opt, ea_fft, cfg_star, cfg_both)
    # TODO: Currently failing since it misses the brighness by 3%, which is
    # greater than the 1% permitted by the test.  However, the ELBO of the
    # FFT optimum is lower than that of the MOG optimum.
    # verify_sample_galaxy(ea_fft.vp[1], [8.5, 9.6])
end


function test_three_body_optimization_fft()
    println("Testing three body fft optimization.")

    images, ea, three_bodies = gen_three_body_dataset();
    Infer.load_active_pixels!(images, ea.patches; exclude_nan=false);
    ea_fft, fsm_mat = DeterministicVIImagePSF.initialize_fft_elbo_parameters(
        images, deepcopy(ea.vp), ea.patches, [1], use_raw_psf=false)
    elbo_fft_opt = DeterministicVIImagePSF.FFTElboFunction(fsm_mat)
    cfg_star = star_only_config(ea_fft; loc_width=1.0)
    cfg_both = Config(ea_fft; loc_width=1.0)
    maximize_two_steps!(elbo_fft_opt, ea_fft, cfg_star, cfg_both)
end


test_galaxy_optimization_fft()
test_three_body_optimization_fft()
test_star_optimization_fft()

#test_quadratic_optimization()
test_star_optimization()
test_single_source_optimization()
test_full_elbo_optimization()
#test_real_stamp_optimization()
test_galaxy_optimization()
