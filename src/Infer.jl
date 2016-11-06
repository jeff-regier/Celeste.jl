"""
We'll delete this module soon, and move most of the these methods to the
`Model` module: the log probability can't be determined without neighbors
and without trimming.
We can't make the move yet, because `PSF` depends on `Transform` still,
and `Transform` depends on `Model` still.
The `infer_source()` method probably belongs in `ParallelRun`, once this
module is deleted.
Currently `infer_source()` only does (deterministic) variational inference.
In the future, `infer_source()` might take a call back function as an
argument, to let it's user run either deterministic VI, stochastic VI,
or MCMC.
"""
module Infer

import WCS

using ..Model
import ..PSF
import ..Log

using ..DeterministicVI
import ..DeterministicVI: load_bvn_mixtures,
                          load_source_brightnesses,
                          get_expected_pixel_brightness!,
                          elbo,
                          maximize_f,
                          clear!



"""
Computes the nearby light sources in the catalog for each of the target
sources.

Arguments:
    target_sources: indexes of astronomical objects in the catalog to infer
    catalog: astronomical objects appearing the images
    images: tiled astronomical images
"""
function find_neighbors(target_sources::Vector{Int64},
                        catalog::Vector{CatalogEntry},
                        images::Vector{TiledImage})
    psf_width_ub = zeros(B)
    for img in images
        psf_width = Model.get_psf_width(img.psf)
        psf_width_ub[img.b] = max(psf_width_ub[img.b], psf_width)
    end

    epsilon_lb = fill(Inf, B)
    for img in images
        Ht, Wt = size(img.tiles)
        epsilon = mean(img.tiles[ceil(Int, Ht/2), ceil(Int, Wt/2)].epsilon_mat)
        epsilon_lb[img.b] = min(epsilon_lb[img.b], epsilon)
    end

    radii_map = zeros(length(catalog))
    for s in 1:length(catalog)
        ce = catalog[s]
        for b in 1:B
            radius_pix = Model.choose_patch_radius(ce, b,
                                                   psf_width_ub[b],
                                                   epsilon_lb[b],
                                                   width_scale=1.2)
            radii_map[s] = max(radii_map[s], radius_pix)
        end
        radii_map[s] = min(radii_map[s], 25) # hack: upper bound radius at 25 arc seconds
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

Arguments:
    images: a collection of (tiled) astronomical images
    neighbors: the other light sources near `entry`
    entry: the source to infer
"""
function infer_source(images::Vector{TiledImage},
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
    @assert length(ea.active_pixels) > 0
    f_evals, max_f, max_x, nm_result = maximize_f(elbo, ea)
    vp[1], max_f
end


"""
For tile of each image, compute a list of the indexes of the catalog entries
that may be relevant to determining the likelihood of that tile.
"""
function get_sky_patches(images::Vector{TiledImage},
                         catalog::Vector{CatalogEntry};
                         radius_override_pix=NaN)
    N = length(images)
    S = length(catalog)
    patches = Array(SkyPatch, S, N)

    for i = 1:N
        img = images[i]

        for s=1:S
            world_center = catalog[s].pos
            pixel_center = WCS.world_to_pix(img.wcs, world_center)
            wcs_jacobian = Model.pixel_world_jacobian(img.wcs, pixel_center)
            radius_pix = Model.choose_patch_radius(pixel_center, catalog[s],
                                                   img.psf, img)
            if !isnan(radius_override_pix)
                radius_pix = radius_override_pix
            end

            patches[s, i] = SkyPatch(world_center,
                                     radius_pix,
                                     img.psf,
                                     wcs_jacobian,
                                     pixel_center)
        end
    end

    patches
end


"""
Test if the there are any intersections between two vectors.
"""
function isintersection_sorted(x::AbstractVector, y::AbstractVector)
    nx, ny = length(x), length(y)
    if nx == 0 || ny == 0
        return false
    end
    @inbounds begin
        i = j = 1
        xi, yj = x[1], y[1]
        while true
            if xi < yj
                if i == nx
                    return false
                else
                    i += 1
                    xi = x[i]
                end
            elseif xi > yj
                if j == ny
                    return false
                else
                    j += 1
                    yj = y[j]
                end
            else
                return true
            end
        end
    end
end


"""
Get pixels significantly above background noise.

Arguments:
  ea: The ElboArgs object
  noise_fraction: The proportion of the noise below which we will remove pixels.
  min_radius_pix: A minimum pixel radius to be included.
"""
function load_active_pixels!(ea::ElboArgs{Float64};
                            noise_fraction=0.1,
                            min_radius_pix=8.0)
    ea.active_pixels = ActivePixel[]

    for n = 1:ea.N
        tiles = ea.images[n].tiles

        # TODO: just loop over the tiles/pixels near the active source(s)
        for t in 1:length(tiles)
            tile = tiles[t]

            tile_ctr = (mean(tile.h_range), mean(tile.w_range))
            h_width, w_width = size(tile.pixels)
            tile_diag = (0.5 ^ 2) * (h_width ^ 2 + w_width ^ 2)

            is_close(s) = begin
                patch_ctr = ea.patches[s, n].pixel_center
                patch_dist = (tile_ctr[1] - patch_ctr[1])^2
                              + (tile_ctr[2] - patch_ctr[2])^2
                patch_r = ea.patches[s, n].radius_pix
                patch_dist <= (tile_diag + patch_r)^2 &&
                   (abs(tile_ctr[1] - patch_ctr[1]) < patch_r + 0.5 * h_width) &&
                   (abs(tile_ctr[2] - patch_ctr[2]) < patch_r + 0.5 * w_width)
            end

            if !any(is_close, ea.active_sources)
                continue
            end

            local_active_sources = Int64[]
            local_centers = Vector{Float64}[]
            for s in ea.active_sources
                if is_close(s)
                    push!(local_active_sources, s)

                    patch = ea.patches[s, n]
                    pix_loc = Model.linear_world_to_pix(patch.wcs_jacobian,
                                                        patch.center,
                                                        patch.pixel_center,
                                                        ea.vp[s][ids.u])
                    push!(local_centers, pix_loc)
                end
            end

            # TODO; use log_prob.jl in the Model module to get the
            # get the expected brightness, not variational inference

            star_mcs, gal_mcs = load_bvn_mixtures(ea, tile.b, calculate_derivs=false)
            sbs = load_source_brightnesses(ea, calculate_derivs=false)
            clear!(ea.elbo_vars)
            ea.elbo_vars.calculate_derivs = false
            ea.elbo_vars.calculate_hessian = false

            for h in tile.h_range, w in tile.w_range
                # The pixel location in the rendered image.
                h_im = h - minimum(tile.h_range) + 1
                w_im = w - minimum(tile.w_range) + 1

                # skip masked pixels
                if isnan(tile.pixels[h_im, w_im])
                    continue
                end

                # if this pixel is bright, let's include it
                get_expected_pixel_brightness!(
                    ea.elbo_vars, h_im, w_im, sbs, star_mcs, gal_mcs, tile,
                    ea, local_active_sources, include_epsilon=false)

                expected_sky = tile.epsilon_mat[h_im, w_im]
                if ea.elbo_vars.E_G.v[1] > expected_sky * noise_fraction
                    push!(ea.active_pixels, ActivePixel(n, t, h_im, w_im))
                    continue
                end

                # include pixels that are close, even if they aren't bright
                for pix_loc in local_centers
                    sq_dist = (h - pix_loc[1])^2 + (w - pix_loc[2])^2
                    if sq_dist < min_radius_pix^2
                        push!(ea.active_pixels, ActivePixel(n, t, h_im, w_im))
                        break
                    end
                end
            end
        end
    end
end


end  # module
