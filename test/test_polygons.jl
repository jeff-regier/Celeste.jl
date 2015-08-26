using Celeste
using Base.Test
using SampleData
using CelesteTypes

import SDSS
import Polygons

println("Running Polygons tests.")

function test_ray_crossing()
    # Check a line segment that is hit in one direction.
    p = Float64[0.5, 0.5]
    v1 = Float64[1, -1]
    v2 = Float64[1, 1]
    r = Float64[1, 0]

    @assert Polygons.ray_crossing(p, r, v1, v2)
    @assert Polygons.ray_crossing(p, r, v2, v1)
    @assert !Polygons.ray_crossing(p, -r, v1, v2)
    @assert !Polygons.ray_crossing(p, -r, v2, v1)

    # Check a line segment that is missed in both directions.
    p = Float64[0.5, 0.5]
    v1 = Float64[1, 1]
    v2 = Float64[1, 2]
    r = Float64[1, 0]
    @assert !Polygons.ray_crossing(p, r, v1, v2)
    @assert !Polygons.ray_crossing(p, r, v2, v1)
    @assert !Polygons.ray_crossing(p, -r, v1, v2)
    @assert !Polygons.ray_crossing(p, -r, v2, v1)

    # Check a line segment that intersects a vertex.
    # Expect that intersecting v2 counts as an intersection
    # but intersecting v1 does not.
    p = Float64[0, 2]
    v1 = Float64[1, 1]
    v2 = Float64[1, 2]
    r = Float64[1, 0]
    @assert Polygons.ray_crossing(p, r, v1, v2)
    @assert !Polygons.ray_crossing(p, r, v2, v1)
    @assert !Polygons.ray_crossing(p, -r, v1, v2)
    @assert !Polygons.ray_crossing(p, -r, v2, v1)


    # Check parallel cases.
    # Parallel to an edge does not count as an intersection unless it intersects
    # the second vertex.

    # Parallel to y-axis:
    v1 = Float64[1, -1]
    v2 = Float64[1, 1]
    r = Float64[0, 1]

    p = Float64[0.5, 0.5]
    @assert !Polygons.ray_crossing(p, r, v1, v2)

    p = Float64[1, 0.5]
    @assert Polygons.ray_crossing(p, r, v1, v2)
    @assert !Polygons.ray_crossing(p, r, v2, v1)

    p = v2
    @assert Polygons.ray_crossing(p, r, v1, v2)
    @assert !Polygons.ray_crossing(p, r, v2, v1)

    # Parallel to x-axis:
    v1 = Float64[-1, 1]
    v2 = Float64[1, 1]
    r = Float64[1, 0]

    p = Float64[0.5, 0.5]
    @assert !Polygons.ray_crossing(p, r, v1, v2)

    p = Float64[0.5, 1]
    @assert Polygons.ray_crossing(p, r, v1, v2)
    @assert !Polygons.ray_crossing(p, r, v2, v1)

    p = v2
    @assert Polygons.ray_crossing(p, r, v1, v2)
    @assert !Polygons.ray_crossing(p, r, v2, v1)

    # Not parallel to an axis:
    v1 = Float64[-1, -1]
    v2 = Float64[1, 1]
    r = Float64[1, 1]

    p = Float64[1.5, 0.5]
    @assert !Polygons.ray_crossing(p, r, v1, v2)

    p = Float64[0.5, 0.5]
    @assert Polygons.ray_crossing(p, r, v1, v2)
    @assert !Polygons.ray_crossing(p, r, v2, v1)

    p = v2
    @assert Polygons.ray_crossing(p, r, v1, v2)
    @assert !Polygons.ray_crossing(p, r, v2, v1)
end


function make_rot_mat(theta::Float64)
    [ cos(theta) -sin(theta); sin(theta) cos(theta) ]
end


function test_point_inside_polygon()
    p_in = Float64[0.2, 0.2]
    p_out = Float64[1.2, 1.2]
    poly = Float64[1 1; -1 1; -1 -1; 1 -1]

    r = Float64[0, 1]
    @assert Polygons.point_inside_polygon(p_in, r, poly)
    @assert !Polygons.point_inside_polygon(p_out, r, poly)

    r = Float64[1, 1]
    @assert Polygons.point_inside_polygon(p_in, r, poly)
    @assert !Polygons.point_inside_polygon(p_out, r, poly)

    r = Float64[0.5, 1]
    @assert Polygons.point_inside_polygon(p_in, r, poly)
    @assert !Polygons.point_inside_polygon(p_out, r, poly)

    offset = [4., -2.]
    @assert Polygons.point_inside_polygon(p_in + offset, r,
        broadcast(+, poly, offset'))
    @assert !Polygons.point_inside_polygon(p_out + offset, r,
        broadcast(+, poly, offset'))

    rot = make_rot_mat(pi / 3)
    @assert Polygons.point_inside_polygon(rot * (p_in + offset), r,
        broadcast(+, poly, offset') * rot')
    @assert !Polygons.point_inside_polygon(rot * (p_out + offset), r,
        broadcast(+, poly, offset') * rot')

    # Check on the edges.
    p_in = Float64[1.0, 0.0]
    p_out = Float64[1.0, -2.0]
    poly = Float64[1 1; -1 1; -1 -1; 1 -1]
    r = Float64[0, 1]
    @assert Polygons.point_inside_polygon(p_in, r, poly)
    @assert !Polygons.point_inside_polygon(p_out, r, poly)

    p_out = Float64[1.0, 2.0]
    @assert !Polygons.point_inside_polygon(p_out, r, poly)
end


function test_point_near_polygon_corner()
    p_in = Float64[1.2, 1.3]
    p_out = Float64[1.4, 1.35]
    poly = Float64[1 1; -1 1; -1 -1; 1 -1]
    radius = 0.5

    @assert Polygons.point_near_polygon_corner(p_in, radius, poly)
    @assert !Polygons.point_near_polygon_corner(p_out, radius, poly)

    offset = [4., -2.]
    rot = make_rot_mat(pi / 3)
    @assert Polygons.point_near_polygon_corner(rot * (p_in + offset), radius,
        broadcast(+, poly, offset') * rot')
    @assert !Polygons.point_near_polygon_corner(rot * (p_out + offset), radius,
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

    @assert Polygons.point_near_line_segment(p_in, radius, v1, v2)
    @assert Polygons.point_near_line_segment(p_in, radius, v2, v1)
    @assert !Polygons.point_near_line_segment(p_out, radius, v1, v2)
    @assert !Polygons.point_near_line_segment(p_out, radius, v2, v1)
    @assert !Polygons.point_near_line_segment(p_out2, radius, v1, v2)
    @assert !Polygons.point_near_line_segment(p_out2, radius, v2, v1)

    offset = [4., -2.]
    rot = make_rot_mat(pi / 3)

    p_in_rot = rot * (p_in + offset)
    p_out_rot = rot * (p_out + offset)
    p_out2_rot = rot * (p_out2 + offset)

    v1_rot = rot * (v1 + offset)
    v2_rot = rot * (v2 + offset)
    @assert Polygons.point_near_line_segment(p_in_rot, radius, v1_rot, v2_rot)
    @assert Polygons.point_near_line_segment(p_in_rot, radius, v2_rot, v1_rot)
    @assert !Polygons.point_near_line_segment(p_out_rot, radius, v1_rot, v2_rot)
    @assert !Polygons.point_near_line_segment(p_out_rot, radius, v2_rot, v1_rot)
    @assert !Polygons.point_near_line_segment(p_out2_rot, radius, v1_rot, v2_rot)
    @assert !Polygons.point_near_line_segment(p_out2_rot, radius, v2_rot, v1_rot)
end


test_ray_crossing()
test_point_inside_polygon()
test_point_near_polygon_corner()
test_point_near_line_segment()
