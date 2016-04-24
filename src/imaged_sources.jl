# Routines for observations of light sources in particular images,
# rather than for sources in the abstract, in physical units,
# and rather than for images alone (that's image_model.jl).


"""
Attributes of the patch of sky surrounding a single
celestial object in a single image.

Attributes:
  - center: The approximate source location in world coordinates
  - radius_pix: The width of the influence of the object in pixel coordinates
  - psf: The point spread function in this region of the sky
  - wcs_jacobian: The jacobian of the WCS transform in this region of the
                  sky for each band
  - pixel_center: The pixel location of center in each band.
"""
immutable SkyPatch
    center::Vector{Float64}
    radius_pix::Float64

    psf::Vector{PsfComponent}
    wcs_jacobian::Matrix{Float64}
    pixel_center::Vector{Float64}
end


"""
Initialize a SkyPatch from an existing SkyPatch and a new PSF.
"""
SkyPatch(patch::SkyPatch, psf::Vector{PsfComponent}) =
    SkyPatch(patch.center, patch.radius_pix, psf, patch.wcs_jacobian,
             patch.pixel_center)


"""Centers of patches in pixel coordinates"""
patch_ctrs_pix(patches::Vector{SkyPatch}) = [p.pixel_center for p in patches]


"""Radii of patches in pixel coordinates"""
patch_radii_pix(patches::Vector{SkyPatch}) = [p.radius_pix for p in patches]


"""
A fast function to determine which sources might belong to which tiles.

Args:
  - tiles: A TiledImage
  - patch_ctrs: Vector of length-2 vectors, giving pixel center of each patch.
  - patch_radii_px: Radius of each patch.

Returns:
  - An array (over tiles) of a vector of candidate
    source patches.  If a patch is a candidate, it may be within the patch
    radius of a point in the tile, though it might not.
"""
function local_source_candidates(tile::ImageTile,
                                 patch_ctrs::Vector{Vector{Float64}},
                                 patch_radii_px::Vector{Float64})
    @assert length(patch_ctrs) == length(patch_radii_px)

    ret = Int[]

    # Find the patches that are less than the radius plus diagonal from the
    # center of the tile.  These are candidates for having some
    # overlap with the tile.
    tile_center = (mean(tile.h_range), mean(tile.w_range))
    tile_diag = (0.5 ^ 2) * (tile.h_width ^ 2 + tile.w_width ^ 2)

    for s in 1:length(patch_ctrs)
        patch_dist = (tile_center[1] - patch_ctrs[s][1])^2
                    + (tile_center[2] - patch_ctrs[s][2])^2
        if patch_dist <= (tile_diag + patch_radii_px[s])^2
            push!(ret, s)
        end
    end

    return ret
end

"""
Args:
  - tile: An ImageTile (containing tile coordinates)
  - patch_ctrs: Vector of length-2 vectors, giving pixel center of each patch.
  - patch_radii_px: Radius of each patch.

Returns:
  - A vector of source ids (from 1 to length(patches)) that influence
    pixels in the tile.  A patch influences a tile if
    there is any overlap in their squares of influence.
"""
function get_local_sources(tile::ImageTile,
                           patch_ctrs::Vector{Vector{Float64}},
                           patch_radii_px::Vector{Float64})
    @assert length(patch_ctrs) == length(patch_radii_px)
    tile_sources = Int[]
    tile_ctr = (mean(tile.h_range), mean(tile.w_range))

    for i in eachindex(patch_ctrs)
        patch_ctr = patch_ctrs[i]
        patch_r = patch_radii_px[i]

        # This is a "ball" in the infinity norm.
        if ((abs(tile_ctr[1] - patch_ctr[1]) < patch_r + 0.5 * tile.h_width) &&
            (abs(tile_ctr[2] - patch_ctr[2]) < patch_r + 0.5 * tile.w_width))
            push!(tile_sources, i)
        end
    end

    tile_sources
end


"""
Get the sources associated with each tile in a TiledImage.

Args:
  - tiled_image: A TiledImage
  - patch_ctrs: Vector of length-2 vectors, giving pixel center of each patch.
  - patch_radii_px: Radius of each patch.
Returns:
  - An array (same dimensions as the tiles) of vectors of indices
    into patches indicating which patches are affected by any pixels
    in the tiles.
"""
function get_tiled_image_sources(tiled_image::TiledImage,
                                 patch_ctrs::Vector{Vector{Float64}},
                                 patch_radii_px::Vector{Float64})
    out = similar(tiled_image, Vector{Int})

    HH, WW = size(tiled_image)
    for ww in 1:WW, hh in 1:HH
        cands = local_source_candidates(tiled_image[hh, ww],
                                        patch_ctrs,
                                        patch_radii_px)
        # get indicies in cands that truly overlap the tile.
        idx = get_local_sources(tiled_image[hh, ww],
                                patch_ctrs[cands],
                                patch_radii_px[cands])
        out[hh, ww] = cands[idx]
    end

    return out
end


function get_tiled_image_sources(tiled_image::TiledImage,
                                 patches::Vector{SkyPatch})
    patch_centers = patch_ctrs_pix(patches)
    patch_radii = patch_radii_pix(patches)
    get_tiled_image_sources(tiled_image, patch_centers, patch_radii)
end


"""
Choose a reasonable patch radius based on the catalog.
TODO: Select this by rendering the object and solving an optimization
problem.

Args:
  - pixel_center: The pixel location of the object
  - ce: The catalog entry for the object
  - psf: The psf at this location
  - img: The image
  - width_scale: A multiple of standard deviations to use
  - max_radius: The maximum radius in pixels.

Returns:
  - A radius in pixels chosen from the catalog entry.
"""
function choose_patch_radius(
            pixel_center::Vector{Float64},
            ce::CatalogEntry,
            psf::Array{PsfComponent},
            img::Image;
            width_scale=1.0,
            max_radius=100)

    psf_width = Model.get_psf_width(psf, width_scale=width_scale)

    # The galaxy scale is the point with half the light -- if the light
    # were entirely in a univariate normal, this would be at 0.67 standard
    # deviations.  We are being a bit conservative here.
    obj_width =
      ce.is_star ? psf_width: width_scale * ce.gal_scale / 0.67 + psf_width

    # Get the average sky noise in a rectangle of the width of the psf.
    h_max, w_max = size(img.epsilon_mat)
    h_lim = [Int(floor((pixel_center[1] - obj_width))),
                   Int(ceil((pixel_center[1] + obj_width)))]
    w_lim = [Int(floor((pixel_center[2] - obj_width))),
                   Int(ceil((pixel_center[2] + obj_width)))]
    h_range = max(h_lim[1], 1):min(h_lim[2], h_max)
    w_range = max(w_lim[1], 1):min(w_lim[2], w_max)
    epsilon = mean(img.epsilon_mat[h_range, w_range])

    flux = ce.is_star ? ce.star_fluxes[img.b] : ce.gal_fluxes[img.b]
    @assert flux > 0.

    # Choose enough pixels that the light is either 90% of the light
    # would be captured from a 1d gaussian or 5% of the sky noise,
    # whichever is a larger radius.
    pdf_90 = exp(-0.5 * (1.64)^2) / (sqrt(2pi) * obj_width)
    pdf_target = min(pdf_90, epsilon / (20 * flux))
    rhs = log(pdf_target) + 0.5 * log(2pi) + log(obj_width)
    radius_req = sqrt(-2 * (obj_width ^ 2) * rhs)
    min(radius_req, max_radius)
end


