using Celeste
using Base.Test
using SampleData
using CelesteTypes

import SDSS
import Util
import WCS

function test_local_sources()
    # Coarse test that local_sources gets the right objects.

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
    # Check that a larger blob gets the same number of objects
    # as a smaller blob.  (This is useful to check edge cases of
    # the polygon logic.)

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

    qy = 0
    for ww=1:200,hh=1:200
        tile = ImageTile(hh, ww, big_blob[1])
        if length(ElboDeriv.local_sources(tile, mp)) > 0
            qy += 1
        end
    end

    @test qy == qx
end


function test_local_sources_3()
    # Test local_sources using world coordinates.

    srand(1)
    test_b = 3 # Will test using this band only
    pix_loc = Float64[50., 50.]
    blob0 = SDSS.load_stamp_blob(dat_dir, "164.4311-39.0359")
    body_loc = WCS.pixel_to_world(blob0[test_b].wcs, pix_loc)
    one_body = [sample_ce(body_loc, true),]

    # Get synthetic blobs but with the original world coordinates.
    for b in 1:5 blob0[b].H, blob0[b].W = 100, 100 end
    blob = Synthetic.gen_blob(blob0, one_body)
    for b in 1:5 blob[b].wcs = blob0[b].wcs end

    tile_width = 1
    patch_radius_pix = 5.

    # Get a patch radius in world coordinates by looking at the world diagonals of
    # a pixel square of a certain size.
    world_quad = WCS.pixel_to_world(blob[test_b].wcs,
        [0. 0.; 0. patch_radius_pix; patch_radius_pix 0; patch_radius_pix patch_radius_pix])
    diags = [ world_quad[i, :]' - world_quad[i + 2, :]' for i=1:2 ]
    patch_radius = maximum([sqrt(dot(d, d)) for d in diags])

    mp = ModelInit.cat_init(one_body, patch_radius=patch_radius, tile_width=tile_width)

    # Source should be present
    tile = ImageTile(round(pix_loc[1] / tile_width), round(pix_loc[2] / tile_width), blob[test_b])
    @assert ElboDeriv.local_sources(tile, mp) == [1]

    # Source should not match when you're 1 tile and a half away along the diagonal plus
    # the pixel radius from the center of the tile.
    tile = ImageTile(ceil((pix_loc[1] + 1.5 * tile_width * sqrt(2) + patch_radius_pix) / tile_width),
                     round(pix_loc[2] / tile_width), blob[test_b])
    @assert ElboDeriv.local_sources(tile, mp) == []

    tile = ImageTile(round((pix_loc[1]) / tile_width),
                     ceil((pix_loc[2]  + 1.5 * tile_width * sqrt(2) + patch_radius_pix) / tile_width), blob[test_b])
    @assert ElboDeriv.local_sources(tile, mp) == []
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
        for i in 1:length(1:length(CanonicalParams))
            @test_approx_eq_eps elbo_tiles.d[i, s] elbo.d[i, s] 1e-5
            @test_approx_eq_eps elbo_patches.d[i, s] elbo.d[i, s] 1e-5
        end
    end

    mp4 = ModelInit.cat_init(three_bodies, patch_radius=35., tile_width=10)
    elbo_both = ElboDeriv.elbo(blob, mp4)
    @test_approx_eq_eps elbo_both.v elbo.v 1e-1

    for s in 1:mp.S
        for i in 1:length(1:length(CanonicalParams))
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


function test_util_bvn_cov()
    e_axis = .7
    e_angle = pi/5
    e_scale = 2.

    manual_11 = e_scale^2 * (1 + (e_axis^2 - 1) * (sin(e_angle))^2)
    util_11 = Util.get_bvn_cov(e_axis, e_angle, e_scale)[1,1]
    @test_approx_eq util_11 manual_11

    manual_12 = e_scale^2 * (1 - e_axis^2) * (cos(e_angle)sin(e_angle))
    util_12 = Util.get_bvn_cov(e_axis, e_angle, e_scale)[1,2]
    @test_approx_eq util_12 manual_12

    manual_22 = e_scale^2 * (1 + (e_axis^2 - 1) * (cos(e_angle))^2)
    util_22 = Util.get_bvn_cov(e_axis, e_angle, e_scale)[2,2]
    @test_approx_eq util_22 manual_22
end

function test_ray_crossing()
    # Check a line segment that is hit in one direction.
    p = Float64[0.5, 0.5]
    v1 = Float64[1, -1]
    v2 = Float64[1, 1]
    r = Float64[1, 0]

    @assert WCS.ray_crossing(p, r, v1, v2)
    @assert WCS.ray_crossing(p, r, v2, v1)
    @assert !WCS.ray_crossing(p, -r, v1, v2)
    @assert !WCS.ray_crossing(p, -r, v2, v1)

    # Check a line segment that is missed in both directions.
    p = Float64[0.5, 0.5]
    v1 = Float64[1, 1]
    v2 = Float64[1, 2]
    r = Float64[1, 0]
    @assert !WCS.ray_crossing(p, r, v1, v2)
    @assert !WCS.ray_crossing(p, r, v2, v1)
    @assert !WCS.ray_crossing(p, -r, v1, v2)
    @assert !WCS.ray_crossing(p, -r, v2, v1)

    # Check a line segment that intersects a vertex.
    # Expect that intersecting v2 counts as an intersection
    # but intersecting v1 does not.
    p = Float64[0, 2]
    v1 = Float64[1, 1]
    v2 = Float64[1, 2]
    r = Float64[1, 0]
    @assert WCS.ray_crossing(p, r, v1, v2)
    @assert !WCS.ray_crossing(p, r, v2, v1)
    @assert !WCS.ray_crossing(p, -r, v1, v2)
    @assert !WCS.ray_crossing(p, -r, v2, v1)


    # Check parallel cases.
    # Parallel to an edge does not count as an intersection unless it intersects
    # the second vertex.

    # Parallel to y-axis:
    v1 = Float64[1, -1]
    v2 = Float64[1, 1]
    r = Float64[0, 1]

    p = Float64[0.5, 0.5]
    @assert !WCS.ray_crossing(p, r, v1, v2)

    p = Float64[1, 0.5]
    @assert WCS.ray_crossing(p, r, v1, v2)
    @assert !WCS.ray_crossing(p, r, v2, v1)

    p = v2
    @assert WCS.ray_crossing(p, r, v1, v2)
    @assert !WCS.ray_crossing(p, r, v2, v1)

    # Parallel to x-axis:
    v1 = Float64[-1, 1]
    v2 = Float64[1, 1]
    r = Float64[1, 0]

    p = Float64[0.5, 0.5]
    @assert !WCS.ray_crossing(p, r, v1, v2)

    p = Float64[0.5, 1]
    @assert WCS.ray_crossing(p, r, v1, v2)
    @assert !WCS.ray_crossing(p, r, v2, v1)

    p = v2
    @assert WCS.ray_crossing(p, r, v1, v2)
    @assert !WCS.ray_crossing(p, r, v2, v1)

    # Not parallel to an axis:
    v1 = Float64[-1, -1]
    v2 = Float64[1, 1]
    r = Float64[1, 1]

    p = Float64[1.5, 0.5]
    @assert !WCS.ray_crossing(p, r, v1, v2)

    p = Float64[0.5, 0.5]
    @assert WCS.ray_crossing(p, r, v1, v2)
    @assert !WCS.ray_crossing(p, r, v2, v1)

    p = v2
    @assert WCS.ray_crossing(p, r, v1, v2)
    @assert !WCS.ray_crossing(p, r, v2, v1)
end


function make_rot_mat(theta::Float64)
    [ cos(theta) -sin(theta); sin(theta) cos(theta) ]
end

function test_point_inside_polygon()
    p_in = Float64[0.2, 0.2]
    p_out = Float64[1.2, 1.2]
    poly = Float64[1 1; -1 1; -1 -1; 1 -1]

    r = Float64[0, 1]
    @assert WCS.point_inside_polygon(p_in, r, poly)
    @assert !WCS.point_inside_polygon(p_out, r, poly)

    r = Float64[1, 1]
    @assert WCS.point_inside_polygon(p_in, r, poly)
    @assert !WCS.point_inside_polygon(p_out, r, poly)

    r = Float64[0.5, 1]
    @assert WCS.point_inside_polygon(p_in, r, poly)
    @assert !WCS.point_inside_polygon(p_out, r, poly)

    offset = [4., -2.]
    @assert WCS.point_inside_polygon(p_in + offset, r,
        broadcast(+, poly, offset'))
    @assert !WCS.point_inside_polygon(p_out + offset, r,
        broadcast(+, poly, offset'))

    rot = make_rot_mat(pi / 3)
    @assert WCS.point_inside_polygon(rot * (p_in + offset), r,
        broadcast(+, poly, offset') * rot')
    @assert !WCS.point_inside_polygon(rot * (p_out + offset), r,
        broadcast(+, poly, offset') * rot')

    # Check on the edges.
    p_in = Float64[1.0, 0.0]
    p_out = Float64[1.0, -2.0]
    poly = Float64[1 1; -1 1; -1 -1; 1 -1]
    r = Float64[0, 1]
    @assert WCS.point_inside_polygon(p_in, r, poly)
    @assert !WCS.point_inside_polygon(p_out, r, poly)

    p_out = Float64[1.0, 2.0]
    @assert !WCS.point_inside_polygon(p_out, r, poly)
end


function test_point_near_polygon_corner()
    p_in = Float64[1.2, 1.3]
    p_out = Float64[1.4, 1.35]
    poly = Float64[1 1; -1 1; -1 -1; 1 -1]
    radius = 0.5

    @assert WCS.point_near_polygon_corner(p_in, radius, poly)
    @assert !WCS.point_near_polygon_corner(p_out, radius, poly)

    offset = [4., -2.]
    rot = make_rot_mat(pi / 3)
    @assert WCS.point_near_polygon_corner(rot * (p_in + offset), radius,
        broadcast(+, poly, offset') * rot')
    @assert !WCS.point_near_polygon_corner(rot * (p_out + offset), radius,
        broadcast(+, poly, offset') * rot')

end

function test_point_near_line_segment()
    radius = 1.0
    v1 = Float64[1., 0.]
    v2 = Float64[2., 0.]

    p_in = Float64[1.7, radius * 0.9]
    p_out = Float64[1.7, radius * 1.1]

    # A point must be between the two segment end points.
    p_out2 = Float64[2.1, 0]

    @assert WCS.point_near_line_segment(p_in, radius, v1, v2)
    @assert WCS.point_near_line_segment(p_in, radius, v2, v1)
    @assert !WCS.point_near_line_segment(p_out, radius, v1, v2)
    @assert !WCS.point_near_line_segment(p_out, radius, v2, v1)
    @assert !WCS.point_near_line_segment(p_out2, radius, v1, v2)
    @assert !WCS.point_near_line_segment(p_out2, radius, v2, v1)

    offset = [4., -2.]
    rot = make_rot_mat(pi / 3)

    p_in_rot = rot * (p_in + offset)
    p_out_rot = rot * (p_out + offset)
    p_out2_rot = rot * (p_out2 + offset)

    v1_rot = rot * (v1 + offset)
    v2_rot = rot * (v2 + offset)
    @assert WCS.point_near_line_segment(p_in_rot, radius, v1_rot, v2_rot)
    @assert WCS.point_near_line_segment(p_in_rot, radius, v2_rot, v1_rot)
    @assert !WCS.point_near_line_segment(p_out_rot, radius, v1_rot, v2_rot)
    @assert !WCS.point_near_line_segment(p_out_rot, radius, v2_rot, v1_rot)
    @assert !WCS.point_near_line_segment(p_out2_rot, radius, v1_rot, v2_rot)
    @assert !WCS.point_near_line_segment(p_out2_rot, radius, v2_rot, v1_rot)
end


function test_id_wcs()
    rand_coord = rand(10, 2)
    @assert WCS.pixel_to_world(SDSS.wcs_id, rand_coord) == rand_coord
    @assert WCS.world_to_pixel(SDSS.wcs_id, rand_coord) == rand_coord
end


function test_pixel_deriv_to_world_deriv()
    field_dir = joinpath(dat_dir, "sample_field")
    run_num = "003900"
    camcol_num = "6"
    field_num = "0269"

    # The gain is wrong but it doesn't matter.
    wcs = SDSS.load_raw_field(field_dir, run_num, camcol_num, field_num, 1, 1.0)[7];

    function test_fun(pix_loc::Array{Float64, 1})
        pix_loc[1]^2 + 0.5 * pix_loc[2]
    end

    function test_fun_grad(pix_loc::Array{Float64, 1})
        Float64[2 * pix_loc[1], 0.5 ]
    end

    function test_fun_world(world_loc::Array{Float64, 1}, wcs::WCSLIB.wcsprm)
        pix_loc = WCS.world_to_pixel(wcs, world_loc)
        test_fun(pix_loc)
    end 

    pix_del = 1e-3
    world_del = 1e-9
    pix_loc = Float64[5, 5]
    pix_loc_1 = pix_loc + pix_del * [1, 0]
    pix_loc_2 = pix_loc + pix_del * [0, 1]
    world_loc = WCS.pixel_to_world(wcs, pix_loc)
    world_loc_1 = world_loc + world_del * [1, 0]
    world_loc_2 = world_loc + world_del * [0, 1]

    @test_approx_eq_eps test_fun(pix_loc) test_fun_world(world_loc, wcs) 1e-8

    pix_deriv = test_fun_grad(pix_loc)
    world_deriv = Float64[ (test_fun_world(world_loc_1, wcs) -
                            test_fun_world(world_loc, wcs)) / world_del
                           (test_fun_world(world_loc_2, wcs) -
                            test_fun_world(world_loc, wcs)) / world_del ]

    relative_err = (WCS.pixel_deriv_to_world_deriv(wcs, pix_deriv, pix_loc) -
                    world_deriv) ./ abs(world_deriv)
    @test_approx_eq_eps relative_err [ 0 0 ] 1e-3
end


####################################################

test_util_bvn_cov()
test_sky_noise_estimates()
test_local_sources()
test_local_sources_2()
test_local_sources_3()
test_ray_crossing()
test_point_inside_polygon()
test_point_near_polygon_corner()
test_point_near_line_segment()
test_id_wcs()
test_pixel_deriv_to_world_deriv()