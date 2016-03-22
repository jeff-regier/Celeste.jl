module WCSUtils

import WCS: WCSTransform, world_to_pix, pix_to_world

export pixel_deriv_to_world_deriv, wcs_id

"""
Convert a world location to a 1-indexed pixel location using a linear
approximation to the true transform.

Args:
    - wcs_jacobian: The jacobian of the transform pixel_loc = F(world_loc)
    - world_offset: The world location at which the jacobian is evaluated
    - pix_offset: The pixel location of the world offset
    - worldcoords: The world location to be translated to pixel coordinates.

Returns:
    - The 1-indexed pixel locations in the same shape as the input.
"""
world_to_pix{T <: Number}(wcs_jacobian::Matrix{Float64},
                          world_offset::Vector{Float64},
                          pix_offset::Vector{Float64},
                          worldcoords::VecOrMat{T}) =
    wcs_jacobian * (worldcoords .- world_offset) + pix_offset


"""
A finite-differences approximation to the jacobian of the transform
from pixel to world coordinates.

Args:
 - wcs: The world coordinate system
 - pix_loc: The location at which to evaluate the jacobian
 - pixel_delt: The step size for the finite difference
               approximation in pixel coordinates

Returns:
 - The jacobian of the transform pixel_coord = F(world_coord).  Following the
   standard definition, the pixel coordinates vary across rows and the world
   coordinates across columns.
"""
function pixel_world_jacobian(wcs::WCSTransform, pix_loc::Array{Float64, 1};
	                      pixel_delt=0.5)

    # Choose a step size.
    # Assume that about a half a pixel is a reasonable step size and the
    # directions are about the same.
    world_loc = pix_to_world(wcs, pix_loc)
    world_delt = maximum(abs(
      pix_to_world(wcs, pix_loc + Float64[pixel_delt, pixel_delt]) - world_loc))

    world_loc_1 = world_loc + world_delt * Float64[1, 0]
    world_loc_2 = world_loc + world_delt * Float64[0, 1]

    hcat((world_to_pix(wcs, world_loc_1) - pix_loc) ./ world_delt,
         (world_to_pix(wcs, world_loc_2) - pix_loc) ./ world_delt)
end


"""
Transform a derivative of a scalar function with respect to pixel
coordinates into a derivatve with respect to world coordinates.

Args:
 - wcs: The world coordinate system object
 - df_dpix: The derivative of a scalar function with respect to pixel coordinates
 - pix_loc: The pixel location at which the derivative was taken.
 - pixel_delt: The step size for the finite difference
               approximation in pixel coordinates

Returns:
 - The derivative with respect to world coordinates.
"""
function pixel_deriv_to_world_deriv(
  wcs::WCSTransform, df_dpix::Array{Float64, 1},
  pix_loc::Array{Float64, 1}; pixel_delt=1e-3)
    @assert length(pix_loc) == length(df_dpix) == 2
    trans = pixel_world_jacobian(wcs, pix_loc, pixel_delt=pixel_delt)
    trans' * df_dpix
end


# A world coordinate system where the world and pixel coordinates are the same.
const wcs_id = WCSTransform(2,
                            cd = Float64[1 0; 0 1],
                            ctype = ["none", "none"],
                            crpix = Float64[1, 1],
                            crval = Float64[1, 1]);

end
