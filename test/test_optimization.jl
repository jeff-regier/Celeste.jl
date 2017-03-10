using Base.Test

using Celeste: Model, SensitiveFloats
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


test_star_optimization()
test_single_source_optimization()
test_full_elbo_optimization()
test_galaxy_optimization()
