"""
Utilities for distances between coordinates on the sphere and matching
sets of coordinates.
"""
module Coordinates

import NearestNeighbors: KDTree, knn

"""
    angular_separation(λ_1, ϕ_1, λ_2, ϕ_2)

Angular separatation in *degrees* between two points on the sphere at
longitude λ and latitude ϕ, both given in *degrees*.
"""
function angular_separation(λ_1, ϕ_1, λ_2, ϕ_2)
    Δλ = λ_2 - λ_1
    sin_Δλ = sind(Δλ)
    cos_Δλ = cosd(Δλ)
    sin_ϕ1 = sind(ϕ_1)
    sin_ϕ2 = sind(ϕ_2)
    cos_ϕ1 = cosd(ϕ_1)
    cos_ϕ2 = cosd(ϕ_2)
    return rad2deg(atan2(hypot(cos_ϕ2 * sin_Δλ,
                               cos_ϕ1 * sin_ϕ2 - sin_ϕ1 * cos_ϕ2 * cos_Δλ),
                         sin_ϕ1 * sin_ϕ2 + cos_ϕ1 * cos_ϕ2 * cos_Δλ))
end


"""
convert (lon, lat) -> (x, y, z) unit vector for multiple coordinates
where spherical coordinates are given in *degrees*.
"""
function _spherical_to_cartesian(lon::AbstractVector, lat::AbstractVector)
    length(lon) == length(lat) || error("lengths of `lon` and `lat` must match")

    T = promote_type(eltype(lon), eltype(lat))
    result = Array{T}(3, length(lon))
    for i in eachindex(lon)
        coslat = cosd(lat[i])
        result[1, i] = coslat*cosd(lon[i])  # x
        result[2, i] = coslat*sind(lon[i])  # y
        result[3, i] = sind(lat[i])  # z
    end
    return result
end


"""
    match_coordinates_sky(lon1, lat1, lon2, lat2; nneighbor=1)

For each position in the set of coordinates `(lon1, lat1)`, find the
`nneighbor`th nearest on-sky match in the set of coordinates `(lon2,
lat2)`. Both sets of coordinates are in *degrees*.  Returns the indicies
of each closest match (`Vector{Int}` of length `length(lon1)`) and the
corresponding distances in degrees (`Vector{Float64}`). `nneighbor=2` can be
used to match a catalog to itself.

# Notes

This function correctly accounts for wrap around and behavior near the poles.
Works by transforming spherical coordinates to cartesian and using a KD tree
to efficiently find nearest neighbors.

An alternative approach that *might* be faster would be to use a hierarchical
triangular mesh (HTM) or HEALPix to spatially index coordinates and then only
search nearby indicies for matches.

This function could potentially be incorporated into the SkyCoords
package eventually, but for now it's unclear where it best fits.
"""
function match_coordinates(lon1::AbstractVector, lat1::AbstractVector,
                           lon2::AbstractVector, lat2::AbstractVector;
                           nneighbor::Int=1)
    cartcoords1 = _spherical_to_cartesian(lon1, lat1)
    cartcoords2 = _spherical_to_cartesian(lon2, lat2)
    kdtree = KDTree(cartcoords2; leafsize = 10)
    all_idxs, all_cartdists = knn(kdtree, cartcoords1, nneighbor)

    # extract nneighbor-th index
    idxs = [v[nneighbor] for v in all_idxs]

    # extract nneighbor-th distance and convert from cartesian to spherical
    dists = [2.0 * asind(v[nneighbor] / 2.0) for v in all_cartdists]

    return idxs, dists
end


end  # module
