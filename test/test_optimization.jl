using Base.Test

using Celeste: Model, SensitiveFloats
using Celeste.DeterministicVI.ConstraintTransforms: ParameterConstraint,
                        ConstraintBatch, BoxConstraint, SimplexConstraint
using Celeste.DeterministicVI: ElboArgs
using Celeste.DeterministicVI.ElboMaximize: ElboConfig, maximize!, elbo_optim_options
using Optim

function verify_sample_galaxy(vs, pos)
    @test vs[ids.is_star[2]] >= 0.99

    @test isapprox(vs[ids.pos[1]], pos[1], atol=0.1)
    @test isapprox(vs[ids.pos[2]], pos[2], atol=0.1)

    @test isapprox(vs[ids.gal_axis_ratio] , 0.7, atol=0.05)
    @test isapprox(vs[ids.gal_frac_dev]  , 0.1, atol=0.08)
    @test isapprox(vs[ids.gal_radius_px], 4.0, atol=0.2)

    phi_hat = vs[ids.gal_angle]
    phi_hat -= floor(phi_hat / pi) * pi
    five_deg = 5 * pi/180
    @test isapprox(phi_hat, pi/4, atol=five_deg)

    brightness_hat = exp(vs[ids.flux_loc[2]] + 0.5 * vs[ids.flux_scale[2]])
    @test isapprox(brightness_hat / sample_galaxy_fluxes[3], 1.0, atol=0.05)

    true_colors = log.(sample_galaxy_fluxes[2:5] ./ sample_galaxy_fluxes[1:4])
    for b in 1:4
        @test isapprox(vs[ids.color_mean[b, 2]], true_colors[b], atol=0.2)
    end
end

#########################################################

function test_single_source_optimization()
    ea, vp, catalog = gen_three_body_dataset();

    s = 2
    ea = make_elbo_args(ea.images, catalog, active_source=s, include_kl=false);
    vp_original = deepcopy(vp);

    cfg = ElboConfig(ea, vp; loc_width=1.0)
    maximize!(ea, vp, cfg)

    # Test that it only optimized source s
    @test vp[s] != vp_original[s]
    for other_s in setdiff(1:ea.S, s)
        @test vp[other_s] ≈ vp_original[other_s]
    end
end


function test_galaxy_optimization()
    ea, vp, catalog = gen_sample_galaxy_dataset(; include_kl = false);
    cfg = ElboConfig(ea, vp; loc_width=3.0)
    maximize!(ea, vp, cfg)
    verify_sample_galaxy(vp[1], [8.5, 9.6])
end


function test_full_elbo_optimization()
    ea, vp, catalog = gen_sample_galaxy_dataset(perturb=true);
    cfg = ElboConfig(ea, vp; loc_width=1.0,
                 optim_options=elbo_optim_options(xtol_abs=0.0))
    maximize!(ea, vp, cfg)
    verify_sample_galaxy(vp[1], [8.5, 9.6]);
end

@testset "optimization" begin
    test_single_source_optimization()
    test_full_elbo_optimization()
    test_galaxy_optimization()
end
