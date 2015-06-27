# written by Jeffrey Regier
# jeff [at] stat [dot] berkeley [dot] edu

module Util

VERSION < v"0.4.0-dev" && using Docile

import WCSLIB

export matvec222, logit, inv_logit

function matvec222(mat::Matrix, vec::Vector)
    # x' A x in a slightly more efficient form.
    (mat[1,1] * vec[1] + mat[1,2] * vec[2]) * vec[1] +
            (mat[2,1] * vec[1] + mat[2,2] * vec[2]) * vec[2]
end

function get_bvn_cov(ab::Float64, angle::Float64, scale::Float64)
    # Unpack a rotation-parameterized BVN covariance matrix.
    #
    # Args:
    #   - ab: The ratio of the minor to the major axis.
    #   - angle: Rotation angle (in radians)
    #   - scale: The major axis.
    #
    #  Returns:
    #    The 2x2 covariance matrix parameterized by the inputs.

    #@assert -pi/2 <= angle < pi/2
    @assert 0 < scale
    @assert 0 < ab <= 1.
    cp, sp = cos(angle), sin(angle)
    R = [[cp -sp], [sp cp]]  # rotates
    D = diagm([1., ab])  # shrinks the minor axis
    W = scale * D * R'
    W' * W  # XiXi
end

function inv_logit(x)
    # TODO: bounds checking
    -log(1.0 ./ x - 1)
end

function logit(x)
    # TODO: bounds checking
    1.0 ./ (1.0 + exp(-x))
end

@doc """
Determine whether a ray in direction r from point p
intersects the edge from v1 to v2 in two dimensions.
""" ->
function ray_crossing(p::Array{Float64, 1}, r::Array{Float64, 1},
                      v1::Array{Float64, 1}, v2::Array{Float64, 1})
    @assert length(p) == length(r) == length(v1) == length(v2) == 2

    delta_v = v2 - v1
    int_mat = hcat(r, -delta_v)
    if det(int_mat) == 0
        # If the ray is parallel to an edge, consider it not to be
        # an intersection.
        return false
    else
        sol =  int_mat \ (v1 - p)
        return 0 <= sol[2] < 1 && sol[1] > 0
    end
end
 
@doc """
Use the ray crossing algorithm to determine whether the point p
is inside a convex polygon with corners v[i, :], i =1:number of edges,
using the ray-casting algorithm in direction r.
This assumes the polygon is not self-intersecting, and does not
handle the edge cases that might arise from non-convex shapes.
A point on the edge of a polygon is considered to be outside the polygon.
""" ->
function point_inside_polygon(p, r, v)

    n_edges = size(v, 1)
    @assert length(p) == length(r) == size(v, 2)
    @assert n_edges >= 3

    num_crossings = 0
    for edge=1:n_edges
        if edge < n_edges
            v1 = v[edge, :][:]
            v2 = v[edge + 1, :][:]
        else # edge == n_edges
            # The final edge from the last vertex back to the first.
            v1 = v[edge, :][:]
            v2 = v[1, :][:]
        end
        crossing = ray_crossing(p, r, v1, v2) ? 1: 0
        num_crossings = num_crossings + crossing
    end

    return num_crossings % 2 == 1
end


function point_near_polygon_corner(p, radius, v)
    n_vertices = size(v, 1)
    @assert length(p) == size(v, 2)
    @assert n_vertices >= 3

    r2 = radius ^ 2
    for vertex=1:n_vertices
        delta = p - v[vertex, :][:]
        if dot(delta, delta) < r2
            return true
        end
    end

    return false    
end

function point_near_line_segment(p, radius, v1, v2)
    delta = v2 - v1
    delta = delta / sqrt(dot(delta, delta))

    delta_vp1 = v1 - p
    delta_vp2 = v2 - p

    delta_along1 = dot(delta_vp1, delta) * delta
    delta_along2 = dot(delta_vp2, delta) * delta

    delta_perp = delta_vp2 - delta_along2

    # Check that the point is between the edges of the line segment
    # and no more than radius away.
    return (sqrt(dot(delta_perp, delta_perp)) < radius) &&
           (dot(delta_along1, delta_along2) < 0)
end

function point_near_polygon_edge(p, radius, v)
    n_edges = size(v, 1)
    @assert length(p) == size(v, 2)
    @assert n_edges >= 3

    for edge=1:n_edges
        if edge < n_edges
            v1 = v[edge, :][:]
            v2 = v[edge + 1, :][:]
        else # edge == n_edges
            # The final edge from the last vertex back to the first.
            v1 = v[edge, :][:]
            v2 = v[1, :][:]
        end
        if point_near_line_segment(p, radius, v1, v2)
            return true
        end
    end
    return false
end


function point_within_radius_of_polygon(p, radius, v)
    return (point_near_polygon_corner(p, radius, v) |
            point_inside_polygon(p, Float64[1, 0], v) |
            point_near_polygon_edge(p, radius, v))
end

@doc """
Convert a world location to a 1-indexed pixel location.

Args:
    - wcs: A world coordinate system object
    - world_loc: Either a 2d vector of world coordinates or a matrix
                 where the world coordinates are rows.

Returns:
    - The 1-indexed pixel locations in the same shape as the input.

The frame files seem to use the order (RA, DEC) for world coordinates,
though you should check the CTYPE1 and CTYPE2 header values if in doubt.
""" ->
function world_to_pixel(wcs::WCSLIB.wcsprm, world_loc::Array{Float64})
    single_row = length(size(world_loc)) == 1 
    if single_row
        # Convert to a row vector if it's a single value
        world_loc = world_loc'
    end

    # wcss2p returns 1-indexed pixel locations.
    pix_loc = WCSLIB.wcss2p(wcs, world_loc')

    if single_row
        return pix_loc[:]
    else
        return pix_loc'
    end
end


@doc """
Convert a 1-indexed pixel location to a world location.

Args:
    - wcs: A world coordinate system object
    - pix_loc: Either a 2d vector of pixel coordinates or a matrix
                 where the pixel coordinates are rows.

Returns:
    - The world locations in the same shape as the input. 

The frame files seem to use the order (RA, DEC) for world coordinates,
though you should check the CTYPE1 and CTYPE2 header values if in doubt.
""" ->
function pixel_to_world(wcs::WCSLIB.wcsprm, pix_loc::Array{Float64})
    single_row = length(size(pix_loc)) == 1 
    if single_row
        # Convert to a row vector if it's a single value
        pix_loc = pix_loc'
    end

    # wcsp2s uses 1-indexed pixel locations.
    world_loc = WCSLIB.wcsp2s(wcs, pix_loc')

    if single_row
        return world_loc[:]
    else
        return world_loc'
    end
end


function world_coordinate_names(wcs::WCSLIB.wcsprm)
    [ unsafe_load(wcs.ctype, i) for i=1:2 ]
end

end