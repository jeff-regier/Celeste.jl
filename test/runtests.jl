#!/usr/bin/env julia

using Celeste
using CelesteTypes
using Base.Test

using Distributions

import Synthetic
import SampleData


function test_local_sources()
    srand(1)
    blob0 = SDSS.load_stamp_blob(dat_dir, "164.4311-39.0359")
    for b in 1:5
        blob0[b].H, blob0[b].W = 112, 238
    end

    three_bodies = [
        sample_ce([4.5, 3.6], false),
        sample_ce([60.1, 82.2], true),
        sample_ce([71.3, 100.4], false),
    ]

    blob = Synthetic.gen_blob(blob0, three_bodies)

    mp = ModelInit.cat_init(three_bodies, patch_radius=20., tile_width=1000)
    @test mp.S == 3

    tile = ImageTile(1, 1, blob[3])
    subset1000 = ElboDeriv.local_sources(tile, mp)
    @test subset1000 == [1,2,3]

    mp.tile_width=10

    subset10 = ElboDeriv.local_sources(tile, mp)
    @test subset10 == [1]

    last_tile = ImageTile(11, 24, blob[3])
    last_subset = ElboDeriv.local_sources(last_tile, mp)
    @test length(last_subset) == 0

    pop_tile = ImageTile(7, 9, blob[3])
    pop_subset = ElboDeriv.local_sources(pop_tile, mp)
    @test pop_subset == [2,3]
end


function test_local_sources_2()
    srand(1)
    blob0 = SDSS.load_stamp_blob(dat_dir, "164.4311-39.0359")
    one_body = [sample_ce([50., 50.], true),]

       for b in 1:5 blob0[b].H, blob0[b].W = 100, 100 end
    small_blob = Synthetic.gen_blob(blob0, one_body)

       for b in 1:5 blob0[b].H, blob0[b].W = 400, 400 end
    big_blob = Synthetic.gen_blob(blob0, one_body)

    mp = ModelInit.cat_init(one_body, patch_radius=35., tile_width=2)

    qx = 0
    for ww=1:50,hh=1:50
        tile = ImageTile(hh, ww, small_blob[2])
        if length(ElboDeriv.local_sources(tile, mp)) > 0
            qx += 1
        end
    end

    @test qx == (36 * 2)^2 / 4

    qy = 0
    for ww=1:200,hh=1:200
        tile = ImageTile(hh, ww, big_blob[1])
        if length(ElboDeriv.local_sources(tile, mp)) > 0
            qy += 1
        end
    end

    @test qy == qx
end


function test_tiling()
    srand(1)
    blob0 = SDSS.load_stamp_blob(dat_dir, "164.4311-39.0359")
    for b in 1:5
        blob0[b].H, blob0[b].W = 112, 238
    end
    three_bodies = [
        sample_ce([4.5, 3.6], false),
        sample_ce([60.1, 82.2], true),
        sample_ce([71.3, 100.4], false),
    ]
       blob = Synthetic.gen_blob(blob0, three_bodies)

    mp = ModelInit.cat_init(three_bodies)
    elbo = ElboDeriv.elbo(blob, mp)

    mp2 = ModelInit.cat_init(three_bodies, tile_width=10)
    elbo_tiles = ElboDeriv.elbo(blob, mp2)
    @test_approx_eq_eps elbo_tiles.v elbo.v 1e-5

    mp3 = ModelInit.cat_init(three_bodies, patch_radius=30.)
    elbo_patches = ElboDeriv.elbo(blob, mp3)
    @test_approx_eq_eps elbo_patches.v elbo.v 1e-5

    for s in 1:mp.S
        for i in 1:length(all_params)
            @test_approx_eq_eps elbo_tiles.d[i, s] elbo.d[i, s] 1e-5
            @test_approx_eq_eps elbo_patches.d[i, s] elbo.d[i, s] 1e-5
        end
    end

    mp4 = ModelInit.cat_init(three_bodies, patch_radius=35., tile_width=10)
    elbo_both = ElboDeriv.elbo(blob, mp4)
    @test_approx_eq_eps elbo_both.v elbo.v 1e-1

    for s in 1:mp.S
        for i in 1:length(all_params)
            @test_approx_eq_eps elbo_both.d[i, s] elbo.d[i, s] 1e-1
        end
    end
end


function test_sky_noise_estimates()
    blobs = Array(Blob, 2)
    blobs[1], mp, three_bodies = gen_three_body_dataset()  # synthetic
    blobs[2] = SDSS.load_stamp_blob(dat_dir, "164.4311-39.0359")  # real

    for blob in blobs
        for b in 1:5
            sdss_sky_estimate = blob[b].epsilon * blob[b].iota
            crude_estimate = median(blob[b].pixels)
            @test_approx_eq_eps sdss_sky_estimate / crude_estimate 1. .3
        end
    end
end


function test_coordinates_vp_conversion()
    blob, mp, three_bodies = gen_three_body_dataset()

    xs = OptimizeElbo.vp_to_coordinates(deepcopy(mp.vp), [ids.lambda[:]])
    vp_new = deepcopy(mp.vp)
    OptimizeElbo.coordinates_to_vp!(deepcopy(xs), vp_new, [ids.lambda[:]])

    @test length(xs) + 3 * 2 * (4 + 1) == 
            length(vp_new[1]) * length(vp_new) == 
            length(mp.vp[1]) * length(mp.vp)

    for s in 1:3
        for p in all_params
            @test_approx_eq mp.vp[s][p] vp_new[s][p]
        end
    end
end


function test_util_bvn_cov()
    rho = .7
    phi = pi/5
    sigma = 2.

    manual_11 = sigma^2 * (1 + (rho^2 - 1) * (sin(phi))^2)
    util_11 = Util.get_bvn_cov(rho, phi, sigma)[1,1]
    @test_approx_eq util_11 manual_11

    manual_12 = sigma^2 * (1 - rho^2) * (cos(phi)sin(phi))
    util_12 = Util.get_bvn_cov(rho, phi, sigma)[1,2]
    @test_approx_eq util_12 manual_12

    manual_22 = sigma^2 * (1 + (rho^2 - 1) * (cos(phi))^2)
    util_22 = Util.get_bvn_cov(rho, phi, sigma)[2,2]
    @test_approx_eq util_22 manual_22
end


####################################################

test_util_bvn_cov()
test_sky_noise_estimates()
test_local_sources_2()
test_local_sources()
test_coordinates_vp_conversion()

include("test_elbo_values.jl")
include("test_derivs.jl")
include("test_optimization.jl")

