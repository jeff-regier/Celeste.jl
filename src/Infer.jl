module Infer

import WCS

using ..Model
using ..ElboDeriv
import ..PSF
import ..OptimizeElbo
import ..Log


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
    cat_local = vcat(entry, neighbors)
    vp = Vector{Float64}[init_source(ce) for ce in cat_local]
    patches, tile_source_map = get_tile_source_map(images, cat_local)
    ea = ElboArgs(images, vp, tile_source_map, patches, [1])
    fit_object_psfs!(ea, ea.active_sources)
    trim_source_tiles!(ea)
    OptimizeElbo.maximize_f(ElboDeriv.elbo, ea)
    vp[1]
end


"""
For tile of each image, compute a list of the indexes of the catalog entries
that may be relevant to determining the likelihood of that tile.
"""
function get_tile_source_map(images::Vector{TiledImage},
                             catalog::Vector{CatalogEntry};
                             radius_override=NaN)
    N = length(images)
    S = length(catalog)
    patches = Array(SkyPatch, S, N)
    tile_source_map = Array(Matrix{Vector{Int}}, N)

    for i = 1:N
        img = images[i]

        for s=1:S
            world_center = catalog[s].pos
            pixel_center = WCS.world_to_pix(img.wcs, world_center)
            wcs_jacobian = Model.pixel_world_jacobian(img.wcs, pixel_center)
            radius_pix = Model.choose_patch_radius(pixel_center, catalog[s],
                                                                img.psf, img)
            if !isnan(radius_override)
                radius_pix = radius_override
            end

            patches[s, i] = SkyPatch(world_center,
                                     radius_pix,
                                     img.psf,
                                     wcs_jacobian,
                                     pixel_center)
        end

        # TODO: get rid of these puny methods
        patch_centers = Model.patch_ctrs_pix(patches[:, i])
        patch_radii = Model.patch_radii_pix(patches[:, i])

        tile_source_map[i] = Model.get_sources_per_tile(images[i].tiles,
                                               patch_centers, patch_radii)
    end

    # TODO: don't create patches
    patches, tile_source_map
end


"""
Updates patches in place with fitted psfs for each active source.
"""
function fit_object_psfs!{NumType <: Number}(
                        ea::ElboArgs{NumType}, 
                        target_sources::Vector{Int})
    # Initialize an optimizer
    initial_psf_params = PSF.initialize_psf_params(psf_K, for_test=false)
    psf_transform = PSF.get_psf_transform(initial_psf_params)
    psf_optimizer = PSF.PsfOptimizer(psf_transform, psf_K)

    for i in 1:length(ea.images)
        # Get a starting point in the middle of the image.
        pixel_loc = Float64[ ea.images[i].H / 2.0, ea.images[i].W / 2.0 ]
        raw_central_psf = ea.images[i].raw_psf_comp(pixel_loc[1], pixel_loc[2])
        central_psf, central_psf_params =
            PSF.fit_raw_psf_for_celeste(raw_central_psf,
                                psf_optimizer, initial_psf_params)

        # Get all relevant sources *in this image*
        relevant_sources = Int[]
        for tile_source_map in ea.tile_source_map[i]
            # check if *any* of this tile's sources are a target, and
            # if so, add *all* the tile sources to the output.
            if length(intersect(target_sources, tile_source_map)) > 0
                relevant_sources = union(relevant_sources, tile_source_map)
            end
        end

        for s in relevant_sources
            patch = ea.patches[s, i]
            # Set the starting point at the center's PSF.
            psf, psf_params = PSF.get_source_psf(
                    patch.center, ea.images[i], psf_optimizer, central_psf_params)
            ea.patches[s, i] = SkyPatch(patch, psf)
        end
    end
end


"""
Set any pixels significantly below background noise for the specified source
to NaN.

Arguments:
  ea: The ElboArgs object
  noise_fraction: The proportion of the noise below which we will remove pixels.
  min_radius_pix: A minimum pixel radius to be included.
"""
function trim_source_tiles!(ea::ElboArgs{Float64};
                            noise_fraction=0.1,
                            min_radius_pix=8.0)
    @assert length(ea.active_sources) == 1
    s = ea.active_sources[1]

    images_out = Array(TiledImage, ea.N)

    for i = 1:ea.N
        img = ea.images[i]
        tiles = img.tiles

        tiles_out = Array(ImageTile, size(tiles)...)
        images_out[i] = TiledImage(img.H, img.W, tiles_out, img.tile_width,
                                img.b, img.wcs,
                                img.psf, img.run_num, img.camcol_num, 
                                img.field_num, img.raw_psf_comp)

        patch = ea.patches[s, i]
        pix_loc = Model.linear_world_to_pix(patch.wcs_jacobian, 
                                            patch.center,
                                            patch.pixel_center,
                                            ea.vp[s][ids.u])

        #TODO: iterate over rows first, not columns
        for hh=1:size(tiles, 1), ww=1:size(tiles, 2) 
            tile = tiles[hh, ww]

            tile_source_map = ea.tile_source_map[i][hh, ww]
            if s in tile_source_map
                tiles_out[hh, ww] = deepcopy(tile)
                pred_tile_pixels =
                    ElboDeriv.tile_predicted_image(tile, ea, [ s ],
                                                   include_epsilon=false)
                for h in tile.h_range, w in tile.w_range
                    # The pixel location in the rendered image.
                    h_im = h - minimum(tile.h_range) + 1
                    w_im = w - minimum(tile.w_range) + 1

                    bright_pixel = pred_tile_pixels[h_im, w_im] >
                       tile.iota_vec[h_im] * tile.epsilon_mat[h_im, w_im] * noise_fraction
                    close_pixel =
                        (h - pix_loc[1]) ^ 2 + (w - pix_loc[2])^2 < min_radius_pix^2

                    if !(bright_pixel || close_pixel)
                        tiles_out[hh, ww].pixels[h_im, w_im] = NaN
                    end
                end
            else
                # TODO: Make tiles simply a vector
                # rather than a 2d array to avoid this hack.
                tiles_out[hh, ww] = ImageTile(i, tile.h_range, tile.w_range,
                                             Array(Float64, 0, 0),
                                             Array(Float64, 0, 0),
                                             Array(Float64, 0))
            end
        end
    end

    # Note: We're changing the images ea points to---the original images aren't
    # mutated
    ea.images = images_out
end


end
