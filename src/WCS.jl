module WCS

VERSION < v"0.4.0-dev" && using Docile

import WCSLIB
import DualNumbers

export sources_near_quadrilateral
export pixel_deriv_to_world_deriv, world_coordinate_names
export wcs_id

# Functions for determining whether a point is near or in a poylgon.

@doc """
Determine whether a given ray intersects a line segment.

Args:
 - p: A point in 2d space
 - r: A 2d vector.  The direction from p to look for a crossing.
 - v1: The first point of the line segment.
 - v2: The second point of the line segment.

 Returns:
  Whether or not the ray r from point p intersects the line segment
  (v1, v2). Intersecting the vertex v1 counts as an intersection but
  intersecting v2 does not.  A point on the ray itself is considered
  an intersection.
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
            return (delta_v2p[1] == 0.) &&
                   (delta_v2p[2] == 0. || sign(r[2]) == sign(delta_v2p[2]))
        elseif r[2] == 0.
            return (delta_v2p[2] == 0.) &&
                   (delta_v2p[1] == 0. || sign(r[1]) == sign(delta_v2p[1]))
        else
            return (delta_v2p[1] / r[1]) == (delta_v2p[2] / r[2]) &&
                   (sign(delta_v2p[1] / r[1]) >= 0.)
        end
    else
        sol =  int_mat \ (v1 - p)
        return 0 < sol[2] <= 1 && sol[1] > 0
    end
end
 
@doc """
Returns whether a point p is inside the polygon with verices v.

Args:
 - p: A point in 2d space.
 - r: A ray for the ray crossing algorithm.  Any ray should do.
 - v: An (edge - 1) x 2 matrix of polygon corners.  An edge from the last
      row to the first is implicit.

Returns:
	Use the ray crossing algorithm to determine whether the point p
	is inside a convex polygon with corners v[i, :], i =1:number of edges,
	using the ray-casting algorithm in direction r.

This assumes the polygon is not self-intersecting, and does not
handle the edge cases that might arise from non-convex shapes.
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

@doc """
Returns whether a point p is within radius of any of the points in v.

Args:
 - p: A point
 - radius: A radius
 - v: An n x 2 matrix of points

Returns:
 Whether p is radius close to any of the points in v.
""" ->
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

@doc """
Returns whether a point p is within radius of the line
segment from v1 to v2 in a direction perpindicular to the
line segment.

Args:
 - p: A point
 - radius: A radius
 - v1, v2: The endpoints of the line segment.

Returns:
 Whether p is within radius of the line segment (v1, v2)
 in a direction perpindicular to (v1, v2).
""" ->
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


@doc """
Returns whether a point p is near the edges of the polygon with verices v.

Args:
 - p: A point in 2d space.
 - radius: The distance from the edge.
 - v: An (edge - 1) x 2 matrix of polygon corners.  An edge from the last
      row to the first is implicit.

Returns:
 Whether the point p is within radius of any of the edges of the polygon v.
""" ->
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


@doc """
Returns whether a point p is in or within radius of a polygon.

Args:
 - p: A point in 2d space.
 - radius: The distance from the edge.
 - v: An (edge - 1) x 2 matrix of polygon corners.  An edge from the last
      row to the first is implicit.

Returns:
 Whether the point p is in or radius-near the polygon v.
""" ->
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
    world_corners = WCS.pixel_to_world(wcs, pix_corners)
    [ point_within_radius_of_polygon(loc[i, :][:],
    	radius[i], world_corners) for i=1:size(loc, 1)]
end


@doc """
sources_in_quadrilateral for a single loc value.
""" ->
function sources_near_quadrilateral(loc::Array{Float64, 1}, radius::Float64,
                                    pix_corners::Array{Float64, 2}, wcs::WCSLIB.wcsprm)
    bool_vec = sources_in_quadrilateral(loc', [ radius ], pix_corners, wcs)
    bool_vec[1]
end


# The next section deals with WCS-specific things.

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
function pixel_to_world{NumType <: Number}(wcs::WCSLIB.wcsprm, pix_loc::Array{NumType})
    # C must be passed Float64 types.
    pix_loc = convert(Array{Float64}, pix_loc)
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

@doc """
Extract the real part from a 1d or 2d DualNumbers array.
""" ->
function flatten_dual_number_array(world_loc::Array{DualNumbers.Dual})
    if length(size(world_loc)) == 1
        float_world_loc = Float64[ DualNumbers.real(loc) for loc in world_loc ]
    else
        @assert length(size(world_loc)) == 2
        float_world_loc = Float64[ DualNumbers.real(world_loc[i, j]) for
                                   i=1:size(world_loc, 1), j=1:size(world_loc, 2)]
    end
    float_world_loc
end

@doc """
Special handling of the DualNumber type which cannot be passed to C.  Note
that forward differentation cannot be applied to the location for this reason.
""" ->
function world_to_pixel(wcs::WCSLIB.wcsprm, world_loc::Array{DualNumbers.Dual})
    world_to_pixel(wcs, flatten_dual_number_array(world_loc))
end

@doc """
Special handling of the DualNumber type which cannot be passed to C.  Note
that forward differentation cannot be applied to the location for this reason.
""" ->
function pixel_to_world(wcs::WCSLIB.wcsprm, world_loc::Array{DualNumbers.Dual})
    pixel_to_world(wcs, flatten_dual_number_array(world_loc))
end


@doc """
The names of the coordinate axes in a WCS coordinate system.
""" ->
function world_coordinate_names(wcs::WCSLIB.wcsprm)
    [ unsafe_load(wcs.ctype, i) for i=1:2 ]
end

@doc """
A finite-differences approximation to the jacobian of the transform
from pixel to world coordinates.

Args:
 - wcs: The world coordinate system
 - pix_loc: The location at which to evaluate the jacobian
 - world_delt: The step size for the finite difference
               approximation in world coordinates

Returns:
 - The jacobian of the transform pixel_coord = F(world_coord).  Following the
   standard definition, the pixel coordinates vary across rows and the world
   coordinates across columns.
""" ->
function pixel_world_jacobian(wcs::WCSLIB.wcsprm, pix_loc::Array{Float64, 1};
	                          world_delt=1e-3)
    world_loc = pixel_to_world(wcs, pix_loc)
    world_delt = Float64[world_delt, world_delt]

    world_loc_1 = world_loc + world_delt[1] * Float64[1, 0]
    world_loc_2 = world_loc + world_delt[2] * Float64[0, 1]

    hcat((world_to_pixel(wcs, world_loc_1) - pix_loc) ./ world_delt[1],
         (world_to_pixel(wcs, world_loc_2) - pix_loc) ./ world_delt[2])
end


@doc """
Transform a derivative of a scalar function with respect to pixel
coordinates into a derivatve with respect to world coordinates.

Args:
 - wcs: The world coordinate system object
 - df_dpix: The derivative of a scalar function with respect to pixel coordinates
 - pix_loc: The pixel location at which the derivative was taken.
 - world_delt: The step size for the finite difference
               approximation in world coordinates

Returns:
 - The derivative with respect to world coordinates.
""" ->
function pixel_deriv_to_world_deriv(wcs::WCSLIB.wcsprm, df_dpix::Array{Float64, 1},
                                    pix_loc::Array{Float64, 1}; world_delt=1e-3)
    @assert length(pix_loc) == length(df_dpix) == 2
    trans = pixel_world_jacobian(wcs, pix_loc, world_delt=world_delt)
    trans' * df_dpix
end

# A world coordinate system where the world and pixel coordinates are the same.
const wcs_id = WCSLIB.wcsprm(2,
                             cd = Float64[1 0; 0 1],
                             ctype = ["none", "none"],
                             crpix = Float64[1, 1],
                             crval = Float64[1, 1]);

end