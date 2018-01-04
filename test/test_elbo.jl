using Celeste: DeterministicVI, SensitiveFloats
import DeterministicVI: calculate_G_s!, load_source_brightnesses, elbo_likelihood

using Base.Test
using Distributions

import ForwardDiff.Dual

import SampleData: gen_two_body_dataset, true_star_init

@testset "elbo" begin

@testset "calculate_G_s overwrites E_G_s, E_G2_s, and var_G_s" begin
    ea, vp, catalog = gen_two_body_dataset()
    elbo_vars = DeterministicVI.ElboIntermediateVariables(Float64, ea.Sa, true, true)

    # let's write some non-trivial derivatives to the fs0m and fs1m
    elbo_likelihood(ea, vp, elbo_vars)

    sbs = load_source_brightnesses(ea, vp)

    # for this call E_G_s and E_G2_s are initialized to zero
    calculate_G_s!(vp, elbo_vars, sbs[2], 4, 2, true)
    ev_cleared = deepcopy(elbo_vars)

    # for this call E_G_s and E_G2_s have not been zeroed out
    calculate_G_s!(vp, elbo_vars, sbs[1], 2, 1, true)
    calculate_G_s!(vp, elbo_vars, sbs[2], 3, 2, true)
    calculate_G_s!(vp, elbo_vars, sbs[2], 4, 2, true)

    @test ev_cleared.E_G_s.v[] ≈ elbo_vars.E_G_s.v[]
    @test ev_cleared.E_G_s.d[] ≈ elbo_vars.E_G_s.d[]
    @test ev_cleared.E_G_s.h[] ≈ elbo_vars.E_G_s.h[]

    @test ev_cleared.E_G2_s.v[] ≈ elbo_vars.E_G2_s.v[]
    @test ev_cleared.E_G2_s.d[] ≈ elbo_vars.E_G2_s.d[]
    @test ev_cleared.E_G2_s.h[] ≈ elbo_vars.E_G2_s.h[]

    @test ev_cleared.var_G_s.v[] ≈ elbo_vars.var_G_s.v[]
    @test ev_cleared.var_G_s.d[] ≈ elbo_vars.var_G_s.d[]
    @test ev_cleared.var_G_s.h[] ≈ elbo_vars.var_G_s.h[]
end


@testset "test bvn cov" begin
    gal_axis_ratio = .7
    gal_angle = pi/5
    gal_radius_px = 2.

    manual_11 = gal_radius_px^2 * (1 + (gal_axis_ratio^2 - 1) * (sin(gal_angle))^2)
    util_11 = DeterministicVI.get_bvn_cov(gal_axis_ratio, gal_angle, gal_radius_px)[1,1]
    @test util_11 ≈ manual_11

    manual_12 = gal_radius_px^2 * (1 - gal_axis_ratio^2) * (cos(gal_angle)sin(gal_angle))
    util_12 = DeterministicVI.get_bvn_cov(gal_axis_ratio, gal_angle, gal_radius_px)[1,2]
    @test util_12 ≈ manual_12

    manual_22 = gal_radius_px^2 * (1 + (gal_axis_ratio^2 - 1) * (cos(gal_angle))^2)
    util_22 = DeterministicVI.get_bvn_cov(gal_axis_ratio, gal_angle, gal_radius_px)[2,2]
    @test util_22 ≈ manual_22
end


@testset "test active sources" begin
    # Test that the derivatives of the expected brightnesses partition in
    # active_sources.
    ea, vp, catalog = gen_two_body_dataset()

    # for subsequent tests, ensure that the first light source
    # has a radius at least as large as both the height and the
    # width of images, and that its center is within the image
    s = 1
    for n = 1:size(ea.patches, 2)
        p = ea.patches[s, n]
        @test 0.5 <= p.world_center[1] <= 20.5
        @test 0.5 <= p.world_center[2] <= 23.5

        # this patch is huge, the bottom left corner should be the
        # bottom left of the image
        @test p.bitmap_offset == [0, 0]

        # all the pixels should be active pixels
        @test size(p.active_pixel_bitmap) == size(ea.images[n].pixels)
        @test all(p.active_pixel_bitmap)
    end

    n = 5
    p = ea.patches[s, n]

    # lets make the second source have no active pixels
    s2 = 2
    for i in 1:ea.N
        fill!(ea.patches[s2, i].active_pixel_bitmap, false)
    end

    P = length(CanonicalParams)
    ea_no2 = ElboArgs(ea.images, ea.patches, [1, 2])
    elbo_no2 = DeterministicVI.elbo_likelihood(ea_no2, vp)
    @test elbo_no2.d[:, 2] ≈ zeros(P, 1)

    # lets make the second source have only a few active pixels
    # in one band, so it and source 1 don't overlap completely
    p2 = ea.patches[s2, n]
    p2.active_pixel_bitmap[10:11,10:11] = true

    # now on to a main test--active source order shouldn't matter
    ea12 = ElboArgs(ea.images, ea.patches, [1, 2])
    elbo_lik_12 = DeterministicVI.elbo_likelihood(ea12, vp)

    ea21 = ElboArgs(ea.images, ea.patches, [2, 1])
    elbo_lik_21 = DeterministicVI.elbo_likelihood(ea21, vp)
    @test elbo_lik_12.v[] ≈ elbo_lik_21.v[]
    @test elbo_lik_12.d[:,1] ≈ elbo_lik_21.d[:,2]
    @test elbo_lik_12.d[:,2] ≈ elbo_lik_21.d[:,1]

    # next main test--deriviatives for active sources don't
    ea1 = ElboArgs(ea.images, ea.patches, [1,])
    elbo_lik_1 = DeterministicVI.elbo_likelihood(ea1, vp)
    # source 1 includes all of source 2's active pixels
    @test elbo_lik_1.v[] ≈ elbo_lik_12.v[]

    ea2 = ElboArgs(ea.images, ea.patches, [2,])
    elbo_lik_2 = DeterministicVI.elbo_likelihood(ea2, vp)

    @test elbo_lik_12.d[:, 1] ≈ elbo_lik_1.d[:, 1]
    @test elbo_lik_12.d[:, 2] ≈ elbo_lik_2.d[:, 1]

    @test elbo_lik_12.h[1:P, 1:P] ≈ elbo_lik_1.h
    @test elbo_lik_12.h[(1:P) + P, (1:P) + P] ≈ elbo_lik_2.h
end

@testset "star truth is most likely" begin
    ea, vp, catalog = true_star_init()

    best = DeterministicVI.elbo_likelihood(ea, vp)

    for bad_a in [.3, .5, .9]
        vp_a = deepcopy(vp)
        vp_a[1][ids.is_star] = [ 1.0 - bad_a, bad_a ]
        bad_a_lik = DeterministicVI.elbo_likelihood(ea, vp_a)
        @test best.v[] > bad_a_lik.v[]
    end

    for h2 in -2:2
        for w2 in -2:2
            if !(h2 == 0 && w2 == 0)
                vp_mu = deepcopy(vp)
                vp_mu[1][ids.pos] += [h2 * .5, w2 * .5]
                bad_mu = DeterministicVI.elbo_likelihood(ea, vp_mu)
                @test best.v[] > bad_mu.v[]
            end
        end
    end

    for delta in [.7, .9, 1.1, 1.3]
        vp_r1 = deepcopy(vp)
        vp_r1[1][ids.flux_loc] += log(delta)
        bad_r1 = DeterministicVI.elbo_likelihood(ea, vp_r1)
        @test best.v[] > bad_r1.v[]
    end

    for b in 1:4
        for delta in [-.3, .3]
            vp_c1 = deepcopy(vp)
            vp_c1[1][ids.color_mean[b, 1]] += delta
            bad_c1 = DeterministicVI.elbo_likelihood(ea, vp_c1)
            @test best.v[] > bad_c1.v[]
        end
    end
end


@testset "galaxy truth is most likely" begin
    ea, vp, catalog = gen_sample_galaxy_dataset(perturb=false)
    vp[1][ids.is_star] = [ 0.01, .99 ]
    best = DeterministicVI.elbo_likelihood(ea, vp)

    for bad_a in [.3, .5, .9]
        vp_a = deepcopy(vp)
        vp_a[1][ids.is_star] = [ 1.0 - bad_a, bad_a ]
        bad_a = DeterministicVI.elbo_likelihood(ea, vp_a)
        @test best.v[] > bad_a.v[];
    end

    for h2 in -2:2
        for w2 in -2:2
            if !(h2 == 0 && w2 == 0)
                vp_mu = deepcopy(vp)
                vp_mu[1][ids.pos] += [h2 * .5, w2 * .5]
                bad_mu = DeterministicVI.elbo_likelihood(ea, vp_mu)
                @test best.v[] > bad_mu.v[]
            end
        end
    end

    for bad_scale in [.8, 1.2]
        vp_r1 = deepcopy(vp)
        vp_r1[1][ids.flux_loc] += 2 * log(bad_scale)
        bad_r1 = DeterministicVI.elbo_likelihood(ea, vp_r1)
        @test best.v[] > bad_r1.v[]
    end

    for n in [:gal_axis_ratio, :gal_angle, :gal_radius_px]
        for bad_scale in [.8, 1.2]
            vp_bad = deepcopy(vp)
            vp_bad[1][getfield(ids, n)] *= bad_scale
            bad_elbo = DeterministicVI.elbo_likelihood(ea, vp_bad)
            @test best.v[] > bad_elbo.v[]
        end
    end

    for b in 1:4
        for delta in [-.3, .3]
            vp_c1 = deepcopy(vp)
            vp_c1[1][ids.color_mean[b, 2]] += delta
            bad_c1 = DeterministicVI.elbo_likelihood(ea, vp_c1)
            @test best.v[] > bad_c1.v[]
        end
    end
end


@testset "manual elbo gradient matches auto diff gradient" begin
    ea, vp, catalog = gen_two_body_dataset()

    # compute the gradient (and the value) of the elbo manually
    elbo_float = DeterministicVI.elbo(ea, vp)

    # create elbo arguments of the Dual number type, that compute the gradient but not
    # the hessian (it doesn't matter that the "perturbation" are all zero, it's just
    # for testing the speed and verifying that it works)
    vp_dual = convert(VariationalParams{Dual{1, Float64}}, vp)

    for s in 1:2, i in 1:length(ids)
        vp_dual[s][i] += ForwardDiff.Dual(0, 1)
        elbo_dual = DeterministicVI.elbo(ea, vp_dual)
        vp_dual[s][i] -= ForwardDiff.Dual(0, 1)

        fwd_diff = elbo_dual.v[].partials[]
        manual_diff = elbo_float.d[i, s]
        @test manual_diff ≈ fwd_diff
    end
end


@testset "manual elbo hessian column sums match auto diff" begin
    ea, vp, catalog = gen_two_body_dataset()

    # compute the hessian of the elbo manually
    elbo_float = DeterministicVI.elbo(ea, vp)

    # create elbo arguments of the Dual number type, that compute the gradient but not
    # the hessian (it doesn't matter that the "perturbation" are all zero, it's just
    # for testing the speed and verifying that it works)
    vp_dual = convert(VariationalParams{Dual{1, Float64}}, vp)

    for s in 1:2, i in 1:length(ids)
        # we're effectively multiplying the hessian by the 1's vector here.
        vp_dual[s][i] += ForwardDiff.Dual(0, 1)
    end

    elbo_dual = DeterministicVI.elbo(ea, vp_dual)

    for s in 1:2, p in 1:length(ids)
        auto_hessian_column_sum = elbo_dual.d[p, s].partials[]
        col_id = (s - 1) * length(ids) + p
        manual_hessian_column_sum = sum(elbo_float.h[:, col_id])
        @test manual_hessian_column_sum ≈ auto_hessian_column_sum
    end
end


@testset "automatic hessian vector product matches finite diff" begin
    ea, vp, catalog = gen_two_body_dataset()

    # compute the gradient (and the value) of the elbo manually
    d0 = DeterministicVI.elbo(ea, vp).d[:]

    vp1 = deepcopy(vp)
    perturbation = 1e-5
    vp1[1] += perturbation
    vp1[2] += perturbation
    d1 = DeterministicVI.elbo(ea, vp1).d[:]

    hv_manual = (d1 - d0) / perturbation

    vp_dual = convert(VariationalParams{Dual{1, Float64}}, vp)

    for s in 1:2, i in 1:length(ids)
        vp_dual[s][i] += ForwardDiff.Dual(0, 1)
    end

    elbo_dual = DeterministicVI.elbo(ea, vp_dual)

    P = length(ids)
    hv_auto = [elbo_dual.d[i].partials[] for i in 1:(2P)]

    for i in 1:20
        @test hv_manual[i] ≈ hv_auto[i] atol=abs(0.01 * hv_auto[i])
    end
end

end
