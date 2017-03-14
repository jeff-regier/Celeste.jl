"""
Routines for single-node inference that aren't specific
to any particular method of inference (e.g MCMC, DeterministicVI),
yet are also not about the statistical model (i.e., the Model
module).
Currently what's in here are routines that effectively
truncate the Gaussians in the model.
TODO: rename this module to something more meaningful
"""
module Infer

import WCS
using StaticArrays

import ..Configs
using ..Model
import ..Log


"""
Computes the nearby light sources in the catalog for each of the target
sources.

Arguments:
    target_sources: indexes of astronomical objects in the catalog to infer
    catalog: astronomical objects appearing the images
    images: astronomical images
"""
function find_neighbors(target_sources::Vector{Int64},
                        catalog::Vector{CatalogEntry},
                        images::Vector{Image})
    psf_width_ub = zeros(B)
    for img in images
        psf_width = Model.get_psf_width(img.psf)
        psf_width_ub[img.b] = max(psf_width_ub[img.b], psf_width)
    end

    epsilon_lb = fill(Inf, B)
    for img in images
        epsilon = img.sky[div(img.H, 2), div(img.W, 2)]
        epsilon_lb[img.b] = min(epsilon_lb[img.b], epsilon)
    end

    radii_map = zeros(length(catalog))
    for s in 1:length(catalog)
        ce = catalog[s]
        for img in images
            radius_pix = Model.choose_patch_radius(ce, img, width_scale=1.2)
            radii_map[s] = max(radii_map[s], radius_pix)
        end
        @assert radii_map[s] <= 25
    end

    dist(ra1, dec1, ra2, dec2) = (3600 / 0.396) * max(abs(dec2 - dec1), abs(ra2 - ra1))

    neighbor_map = Vector{Int64}[Int64[] for s in target_sources]

    # If this loop isn't super fast in pratice, we can tile (the sky, not the
    # images) or build a spatial index with a library before distributing
    for ts in 1:length(target_sources)
        s = target_sources[ts]
        ce = catalog[s]

        for s2 in 1:length(catalog)
            ce2 = catalog[s2]
            ctrs_dist = dist(ce.pos[1], ce.pos[2], ce2.pos[1], ce2.pos[2])

            if s2 != s && ctrs_dist < radii_map[s] + radii_map[s2]
                push!(neighbor_map[ts], s2)
            end
        end
    end

    neighbor_map
end


function get_sky_patches(images::Vector{Image},
                         catalog::Vector{CatalogEntry};
                         radius_override_pix=NaN)
    N = length(images)
    S = length(catalog)
    patches = Matrix{SkyPatch}(S, N)

    for n = 1:N
        img = images[n]

        for s=1:S
            world_center = catalog[s].pos
            pixel_center = WCS.world_to_pix(img.wcs, world_center)
            wcs_jacobian = Model.pixel_world_jacobian(img.wcs, pixel_center)
            radius_pix = Model.choose_patch_radius(catalog[s], img, width_scale=1.2)
            @assert radius_pix <= 25
            if !isnan(radius_override_pix)
                radius_pix = radius_override_pix
            end

            hmin = max(0, floor(Int, pixel_center[1] - radius_pix - 1))
            hmax = min(img.H - 1, ceil(Int, pixel_center[1] + radius_pix - 1))
            wmin = max(0, floor(Int, pixel_center[2] - radius_pix - 1))
            wmax = min(img.W - 1, ceil(Int, pixel_center[2] + radius_pix - 1))

            # some light sources are so far from some images that they don't
            # overlap at all
            H2 = max(0, hmax - hmin + 1)
            W2 = max(0, wmax - wmin + 1)

            # all pixels are active by default
            active_pixel_bitmap = trues(H2, W2)

            patches[s, n] = SkyPatch(world_center,
                                     radius_pix,
                                     img.psf,
                                     wcs_jacobian,
                                     pixel_center,
                                     SVector(hmin, wmin),
                                     active_pixel_bitmap)
        end
    end

    patches
end


"""
Record which pixels in each patch will be considered when computing the
objective function.

Non-standard arguments:
  noise_fraction: The proportion of the noise below which we will remove pixels.
"""
function load_active_pixels!(config::Configs.Config,
                             images::Vector{Image},
                             patches::Matrix{SkyPatch};
                             exclude_nan=true,
                             noise_fraction=0.5)
    S, N = size(patches)

    for n = 1:N, s=1:S
        img = images[n]
        p = patches[s,n]

        # (h2, w2) index the local patch, while (h, w) index the image
        H2, W2 = size(p.active_pixel_bitmap)
        for w2 in 1:W2, h2 in 1:H2
            h = p.bitmap_offset[1] + h2
            w = p.bitmap_offset[2] + w2

            # skip masked pixels
            if isnan(img.pixels[h, w]) && exclude_nan
                p.active_pixel_bitmap[h2, w2] = false
                continue
            end

            # include pixels that are close, even if they aren't bright
            sq_dist = (h - p.pixel_center[1])^2 + (w - p.pixel_center[2])^2
            if sq_dist < config.min_radius_pix^2
                p.active_pixel_bitmap[h2, w2] = true
                continue
            end

            # if this pixel is bright, let's include it
            # (in the future we may want to do something fancier, like
            # fitting an elipse, so we don't include nearby sources' pixels,
            # or adjusting active pixels during the optimization)
            # Note: This is risky because bright pixels are disproportionately likely
            # to get included, even if it's because of noise. Therefore it's important
            # to keep the noise fraction pretty low.
            threshold = img.iota_vec[h] * img.sky[h, w] * (1. + noise_fraction)
            p.active_pixel_bitmap[h2, w2] = img.pixels[h, w] > threshold
        end
    end
end

# legacy wrapper
function load_active_pixels!(images::Vector{Image},
                             patches::Matrix{SkyPatch};
                             exclude_nan=true,
                             noise_fraction=0.5)
    load_active_pixels!(
        Configs.Config(),
        images,
        patches,
        exclude_nan=exclude_nan,
        noise_fraction=noise_fraction,
    )
end

# The range of image pixels in a vector of patches
function get_active_pixel_range(
    patches::Matrix{SkyPatch}, sources::Vector{Int}, n::Int)
    H_min = minimum([ p.bitmap_offset[1] + 1 for p in patches[sources, n] ])
    W_min = minimum([ p.bitmap_offset[2] + 1 for p in patches[sources, n] ])
    H_max = maximum([ p.bitmap_offset[1] + size(p.active_pixel_bitmap, 1)
                      for p in patches[sources, n] ])
    W_max = maximum([ p.bitmap_offset[2] + size(p.active_pixel_bitmap, 2)
                      for p in patches[sources, n] ])
    H_min, W_min, H_max, W_max
end


# Is a pixel (h, w) in whole-image coordinates an active pixel in the patch p?
function is_pixel_in_patch(h::Int, w::Int, p::SkyPatch)
    hp = h - p.bitmap_offset[1]
    wp = w - p.bitmap_offset[2]
    in_patch =
        (hp > 0) &
        (wp > 0) &
        (hp <= size(p.active_pixel_bitmap, 1)) &
        (wp <= size(p.active_pixel_bitmap, 2))
    if !in_patch
        return false
    else
        return p.active_pixel_bitmap[hp, wp]
    end
end



end  # module
