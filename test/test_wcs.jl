using Celeste
using Base.Test
using SampleData
using CelesteTypes

import SDSS
import WCS


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
    @assert WCS.pixel_to_world(WCS.wcs_id, rand_coord) == rand_coord
    @assert WCS.world_to_pixel(WCS.wcs_id, rand_coord) == rand_coord
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


function test_world_to_pixel()
    field_dir = joinpath(dat_dir, "sample_field")
    run_num = "003900"
    camcol_num = "6"
    field_num = "0269"

    # The gain will not be used.
    nelec, calib_col, sky_grid, sky_x, sky_y, sky_image, wcs =
        SDSS.load_raw_field(field_dir, run_num, camcol_num, field_num, 3, 1.0);

    pix_center = Float64[0.5 * size(nelec, 1), 0.5 * size(nelec, 1)]
    pix_loc = pix_center + [5., 3.]
    world_center = WCS.pixel_to_world(wcs, pix_center)
    world_loc = WCS.pixel_to_world(wcs, pix_loc)

    wcs_jacobian = WCS.pixel_world_jacobian(wcs, pix_center);

    pix_loc_test1 = WCS.world_to_pixel(wcs, world_loc)
    pix_loc_test2 = WCS.world_to_pixel(wcs_jacobian, world_center, pix_center, world_loc)

    # Note that the accuracy of the linear approximation isn't great.
    @test_approx_eq(pix_loc_test1, pix_loc)
    @test_approx_eq_eps(pix_loc_test2, pix_loc, 1e-2)
end


test_ray_crossing()
test_point_inside_polygon()
test_point_near_polygon_corner()
test_point_near_line_segment()
test_id_wcs()
test_pixel_deriv_to_world_deriv()
test_world_to_pixel()