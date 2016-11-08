module Infer

import WCS
using StaticArrays

using ..Model
import ..PSF
import ..Log

using ..DeterministicVI
import ..DeterministicVI: elbo, maximize_f


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
        epsilon = mean(img.epsilon_mat)  # use just the center if this is slow
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

    # compute distance in pixels using small-distance approximation
    dist(ra1, dec1, ra2, dec2) = (3600 / 0.396) * (sqrt((dec2 - dec1)^2 +
                                  (cos(dec1) * (ra2 - ra1))^2))

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


"""
Infers one light source. This routine is intended to be called in parallel,
once per target light source.

Currently `infer_source()` only does (deterministic) variational inference.
In the future, `infer_source()` might take a call back function as an
argument, to let it's user run either deterministic VI, stochastic VI,
or MCMC.

Arguments:
    images: a collection of astronomical images
    neighbors: the other light sources near `entry`
    entry: the source to infer
"""
function infer_source(images::Vector{Image},
                      neighbors::Vector{CatalogEntry},
                      entry::CatalogEntry)
    if length(neighbors) > 100
        Log.warn("Excessive number ($(length(neighbors))) of neighbors")
    end
    cat_local = vcat(entry, neighbors)
    vp = Vector{Float64}[init_source(ce) for ce in cat_local]
    patches = get_sky_patches(images, cat_local)
    ea = ElboArgs(images, vp, patches, [1])
    load_active_pixels!(ea)
    f_evals, max_f, max_x, nm_result = maximize_f(elbo, ea)
    vp[1], max_f
end

"""
  noise_fraction: The proportion of the noise below which we will remove pixels.
  min_radius_pix: A minimum pixel radius to be included.
"""
function get_sky_patches(images::Vector{Image},
                         catalog::Vector{CatalogEntry};
                         radius_override_pix=NaN,
                         noise_fraction=0.1,
                         min_radius_pix=8.0)
    N = length(images)
    S = length(catalog)
    patches = Array(SkyPatch, S, N)

    for n = 1:N
        img = images[n]

        for s=1:S
            world_center = catalog[s].pos
            pixel_center = WCS.world_to_pix(img.wcs, world_center)
            wcs_jacobian = Model.pixel_world_jacobian(img.wcs, pixel_center)
            radius_pix = Model.choose_patch_radius(catalog[s], img, width_scale=1.2)
            if !isnan(radius_override_pix)
                radius_pix = radius_override_pix
            end

            center_int = round(Int, pixel_center)
            radius_int = ceil(Int, radius_pix)
            hmin = max(1, center_int[1] - radius_int)
            hmax = min(img.H, center_int[1] + radius_int)
            wmin = max(1, center_int[2] - radius_int)
            wmax = min(img.W, center_int[2] + radius_int)

            # some light sources are so far from some images that they don't
            # overlap at all
            H2 = max(0, hmax - hmin)
            W2 = max(0, wmax - wmin)

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
Get pixels significantly above background noise.

Arguments:
  ea: The ElboArgs object
  noise_fraction: The proportion of the noise below which we will remove pixels.
  min_radius_pix: A minimum pixel radius to be included.
"""
function load_active_pixels!(ea::ElboArgs{Float64};
                            noise_fraction=0.2,
                            min_radius_pix=8.0)
    for n = 1:ea.N, s=1:ea.S
        img = ea.images[n]
        p = ea.patches[s,n]

        # (h2, w2) index the local patch, while (h, w) index the image
        H2, W2 = size(p.active_pixel_bitmap)
        for w2 in 1:W2, h2 in 1:H2
            h = p.bitmap_corner[1] + h2
            w = p.bitmap_corner[2] + w2

            # skip masked pixels
            if isnan(img.pixels[h, w])
                p.active_pixel_bitmap[h2, w2] = false
                continue
            end

            # include pixels that are close, even if they aren't bright
            sq_dist = (h - p.pixel_center[1])^2 + (w - p.pixel_center[2])^2
            if sq_dist < min_radius_pix^2
                p.active_pixel_bitmap[h2, w2] = true
                continue
            end

            # if this pixel is bright, let's include it
            # (in the future we may want to do something fancier, like
            # fitting an elipse, so we don't include nearby sources' pixels,
            # or adjusting active pixels during the optimization)
            threshold = img.epsilon_mat[h, w] * (1. + noise_fraction)
            p.active_pixel_bitmap[h2, w2] = img.pixels[h, w] > threshold
        end
    end
end


end  # module
