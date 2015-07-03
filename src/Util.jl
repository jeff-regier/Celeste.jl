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


# TODO: make a WCS.jl file.

@doc """
Determine whether a ray in direction r from point p
intersects the edge from v1 to v2 in two dimensions.
Intersecting the vertex v1 counts as an intersection but
intersecting v2 does not.
""" ->
function ray_crossing(p::Array{Float64, 1}, r::Array{Float64, 1},
                      v1::Array{Float64, 1}, v2::Array{Float64, 1})
    @assert length(p) == length(r) == length(v1) == length(v2) == 2

    delta_v = v2 - v1
    int_mat = hcat(r, -delta_v)
    if det(int_mat) == 0
        # If the ray is parallel to an edge, consider it to be
        # an intersection only if it passes through v2.
        delta_v2p = v2 - p
        if r[1] == 0.
            return (delta_v2p[1] == 0.) && (delta_v2p[2] == 0. || sign(r[2]) == sign(delta_v2p[2]))
        elseif r[2] == 0.
            return (delta_v2p[2] == 0.) && (delta_v2p[1] == 0. || sign(r[1]) == sign(delta_v2p[1]))
        else
            return (delta_v2p[1] / r[1]) == (delta_v2p[2] / r[2]) && (sign(delta_v2p[1] / r[1]) >= 0.)
        end
    else
        sol =  int_mat \ (v1 - p)
        return 0 < sol[2] <= 1 && sol[1] > 0
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
Check whether a set of sources is within <radius> world coordinates of
a quadrilateral defined in pixel coordinates.

Args:
  - loc: An S x 2 array of source locations in world coordinates,
         with the locations as row vectors.
  - radius: An array of radii in world coordinates, one for each loc.
  - pix_corners: A 4 x 2 array of quadrilateral corners in pixel coordinates.
                 The corners must make a quadrilateral when traced in order of the
                 rows with a final edge between the last row and the first.

Returns:
    An array of booleans for whether each row of loc is within radius of the
    pix_corners quadrilateral.
""" ->
function sources_near_quadrilateral(loc::Array{Float64, 2}, radius::Array{Float64, 1},
                                    pix_corners::Array{Float64, 2}, wcs::WCSLIB.wcsprm)
    @assert size(loc, 2) == size(pix_corners, 2) == 2
    @assert size(radius, 1) == size(loc, 1)
    world_corners = Util.pixel_to_world(wcs, pix_corners)
    [ Util.point_within_radius_of_polygon(loc[i, :][:], radius[i], world_corners) for i=1:size(loc, 1)]
end


@doc """
sources_in_quadrilateral for a single loc value.
""" ->
function sources_near_quadrilateral(loc::Array{Float64, 1}, radius::Float64,
                                    pix_corners::Array{Float64, 2}, wcs::WCSLIB.wcsprm)
    bool_vec = sources_in_quadrilateral(loc', [ radius ], pix_corners, wcs)
    bool_vec[1]
end


################ WCS stuff below

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


@doc """
Transform a derivative of a scalar function with respect to pixel
coordinates into a derivatve with respect to world coordinates.

Arguments:
    - wcs: The world coordinate system object
    - df_dpix: The derivative of a scalar function with respect to pixel coordinates
    - pix_loc: The pixel location at which the derivative was taken.
    - pix_delt: In pixel coordinates, the size of the difference for the finite
                difference approximation of the wcs transform.
""" ->
function pixel_deriv_to_world_deriv(wcs::WCSLIB.wcsprm, df_dpix::Array{Float64, 1},
                                    pix_loc::Array{Float64, 1}; pix_delt=0.1)

    @assert length(pix_loc) == length(df_dpix) == 2

    # TODO: This needs to be much faster if we're going to do it for every source
    # at every pixel.

    # Assume that 0.1 pixels is a resonable step size irrespective of 
    # the world coordinates.  Choose a step size in world coordinates on
    # the same order.
    # world_delt1 = abs(pixel_to_world(wcs, pix_loc + pix_delt * [0, 1]) - world_loc)
    # world_delt2 = abs(pixel_to_world(wcs, pix_loc + pix_delt * [1, 0]) - world_loc)
    # world_delt = Float64[ max(world_delt1[i], world_delt2[i]) for i=1:2 ]
    world_loc = pixel_to_world(wcs, pix_loc)
    world_delt = Float64[1e-3, 1e-3]

    world_loc_1 = world_loc + world_delt[1] * Float64[1, 0]
    world_loc_2 = world_loc + world_delt[2] * Float64[0, 1]

    #transform = vcat((world_to_pixel(wcs, world_loc_1) - pix_loc)' / world_delt[1],
    #                 (world_to_pixel(wcs, world_loc_2) - pix_loc)' / world_delt[2])
    #transform * df_dpix

    Float64[ dot(world_to_pixel(wcs, world_loc_1) - pix_loc, df_dpix / world_delt[1]),
             dot(world_to_pixel(wcs, world_loc_2) - pix_loc, df_dpix / world_delt[2]) ]

end

end