"""
This module contains code that necessary to initialize the model,
but that couldn't go in Model because of dependence on modules 
that import Model (i.e. PSF and ElboDeriv)

Eventually the dependency on ElboDeriv will be severed when we
move to using blob detection algorithms to decide what pixels
are relevant, rather than rendering the image ElboDeriv to
screen pixels. (Eventually we won't have a good initial catalog.)

The PSF module should eventually be included in Model, once
we're learning the PSF outselves, rather than fitting SDSS.
Until then, we have both Model and ModelInit.
"""
module ModelInit

import Logging

using ..Model
import ..WCSUtils
import ..PSF


"""
Initilize the model params to the given catalog and tiled image.

Args:
  - tiled_blob: A TiledBlob
  - blob: The original Blob
  - cat: A vector of catalog entries
  - fit_psf: Whether to give each patch its own local PSF
  - patch_radius: If set less than Inf or if radius_from_cat=false,
                  the radius in world coordinates of each patch.
  - radius_from_cat: If true, choose the patch radius from the catalog.
                     If false, use patch_radius for each patch.
"""
function initialize_model_params(
            tiled_blob::TiledBlob,
            blob::Blob,
            cat::Vector{CatalogEntry};
            fit_psf::Bool=true,
            patch_radius::Float64=NaN)

    @assert length(tiled_blob) == length(blob)
    @assert length(cat) > 0

    Logging.info("Loading variational parameters from catalogs.")

    vp = Array{Float64, 1}[Model.init_source(ce) for ce in cat]
    mp = ModelParams(vp, Model.load_prior())
    mp.objids = ASCIIString[cat_entry.objid for cat_entry in cat]

    mp.patches = Array(SkyPatch, mp.S, length(blob))
    mp.tile_sources = Array(Array{Vector{Int}, 2}, length(blob))

    for b = 1:length(blob)
        img = blob[b]

        for s=1:mp.S
            world_center = cat[s].pos

            pixel_center = WCSUtils.world_to_pix(img.wcs, world_center)
            wcs_jacobian = WCSUtils.pixel_world_jacobian(img.wcs, pixel_center)
            psf = fit_psf ? PSF.get_source_psf(world_center, img)[1] : img.psf

            radius_pix = Model.choose_patch_radius(pixel_center, cat[s], psf, img)

            # for testing
            if !isnan(patch_radius)
                radius_pix = maxabs(eigvals(wcs_jacobian)) * patch_radius
            end

            mp.patches[s, b] = SkyPatch(world_center,
                                        radius_pix,
                                        psf,
                                        wcs_jacobian,
                                        pixel_center)
        end

        mp.tile_sources[b] = Model.get_tiled_image_sources(tiled_blob[b],
                                                           mp.patches[:, b])
    end

    return mp
end


"""
Update ModelParams with the PSFs for a range of object ids.

Args:
  - mp: A ModelParams whose patches will be updated.
  - relevant_sources: A vector of source ids that index into mp.patches
  - blob: A vector of images.

Returns:
  - Updates mp.patches in place with fitted psfs for each source in
    relevant_sources.
"""
function fit_object_psfs!{NumType <: Number}(
    mp::ModelParams{NumType}, target_sources::Vector{Int}, blob::Blob)

    # Initialize an optimizer
    initial_psf_params = PSF.initialize_psf_params(psf_K, for_test=false);
    psf_transform = PSF.get_psf_transform(initial_psf_params);
    psf_optimizer = PSF.PsfOptimizer(psf_transform, psf_K);

    @assert size(mp.patches, 2) == length(blob)

    for b in 1:length(blob)    # loop over images
        Logging.debug("Fitting PSFS for band $b")
        # Get a starting point in the middle of the image.
        pixel_loc = Float64[ blob[b].H / 2.0, blob[b].W / 2.0 ]
        raw_central_psf = blob[b].raw_psf_comp(pixel_loc[1], pixel_loc[2])
        central_psf, central_psf_params =
            PSF.fit_raw_psf_for_celeste(raw_central_psf, psf_optimizer, initial_psf_params)

        # Get all relevant sources *in this image*
        relevant_sources = get_all_relevant_sources_in_image(mp.tile_sources[b],
                                                             target_sources)

        for s in relevant_sources
            Logging.debug("Fitting PSF for b=$b, source=$s, objid=$(mp.objids[s])")
            patch = mp.patches[s, b]
            # Set the starting point at the center's PSF.
            psf, psf_params =
                PSF.get_source_psf(
                    patch.center, blob[b], psf_optimizer, central_psf_params)
            mp.patches[s, b] = SkyPatch(patch, psf)
        end
    end
end


"""
Return an array of source indices that have some overlap with target_s.

Args:
  - mp: The ModelParams
  - target_s: The index of the source of interest

Returns:
  - An array of integers that index into mp.s representing all sources that
    co-occur in at least one tile with target_s, including target_s itself.
"""
function get_relevant_sources{NumType <: Number}(mp::ModelParams{NumType},
                                                 target_s::Int)
    relevant_sources = Int[]
    for b = 1:length(mp.tile_sources), tile_sources in mp.tile_sources[b]
        if target_s in tile_sources
            relevant_sources = union(relevant_sources, tile_sources);
        end
    end

    relevant_sources
end


"""
Return indicies of all sources relevant to any of a set of target sources
in the given image.

# Arguments
  * `mp::ModelParams`: Model parameters.
  * `targets::Vector{Int}`: Indicies of target sources.
  * `b::Int`: Index of image.

# Returns
* `Vector{Int}`: Array of integers that index into mp.s. These represent
  all sources that co-occur in at least one tile with *any* of the sources
  in `targets`.
"""
function get_all_relevant_sources_in_image(
                    sources_by_tile::Matrix{Vector{Int}},
                    target_sources::Vector{Int})
    out = Int[]

    for tile_sources in sources_by_tile  # loop over image tiles
        # check if *any* of this tile's sources are a target, and
        # if so, add *all* the tile sources to the output.
        if length(intersect(target_sources, tile_sources)) > 0
            out = union(out, tile_sources)
        end
    end

    out
end


import ..ElboDeriv

"""
Set any pixels significantly below background noise for the
specified source to NaN.

Arguments:
  s: The source index that we are trimming to
  mp: The ModelParams object
  tiled_blob: The original tiled blob
  noise_fraction: The proportion of the noise below which we will remove pixels.
  min_radius_pix: A minimum pixel radius to be included.

Returns:
  A new TiledBlob.  Tiles that do not contain the source will be pseudo-tiles
  with empty pixel and noise arrays.  Tiles that contain the source will
  be the same as the original tiles but with NaN where the expected source
  electron counts are below <noise_fraction> of the noise at that pixel.
"""
function trim_source_tiles(
        s::Int, mp::ModelParams{Float64}, tiled_blob::TiledBlob;
        noise_fraction::Float64=0.1, min_radius_pix::Float64=8.0)

    trimmed_tiled_blob =
        Array{ImageTile, 2}[ Array(ImageTile, size(tiled_blob[b])...) for
                                                 b=1:length(tiled_blob)];

    min_radius_pix_sq = min_radius_pix ^ 2
    for b = 1:length(tiled_blob)
        Logging.debug("Processing band $b...")

        patch = mp.patches[s, b]
        world_loc = mp.vp[s][ids.u]
        pix_loc = WCSUtils.world_to_pix(patch.wcs_jacobian, 
                                        patch.center,
                                        patch.pixel_center,
                                        world_loc)

        H, W = size(tiled_blob[b])
        @assert size(mp.tile_sources[b]) == size(tiled_blob[b])
        for hh=1:H, ww=1:W
            tile = tiled_blob[b][hh, ww];
            tile_sources = mp.tile_sources[b][hh, ww]
            has_source = s in tile_sources
            bright_pixels = Bool[];
            if has_source
                pred_tile_pixels =
                    ElboDeriv.tile_predicted_image(tile, mp, [ s ],
                                                   include_epsilon=false);
                tile_copy = deepcopy(tiled_blob[b][hh, ww]);

                for h in tile.h_range, w in tile.w_range
                    # The pixel location in the rendered image.
                    h_im = h - minimum(tile.h_range) + 1
                    w_im = w - minimum(tile.w_range) + 1

                    keep_pixel = false
                    bright_pixel = tile.constant_background ?
                        pred_tile_pixels[h_im, w_im] >
                            tile.iota * tile.epsilon * noise_fraction:
                        pred_tile_pixels[h_im, w_im] >
                            tile.iota_vec[h_im] * tile.epsilon_mat[h_im, w_im] * noise_fraction
                    close_pixel =
                        (h - pix_loc[1]) ^ 2 + (w - pix_loc[2]) ^ 2 < min_radius_pix_sq

                    if !(bright_pixel || close_pixel)
                        tile_copy.pixels[h_im, w_im] = NaN
                    end
                end

                trimmed_tiled_blob[b][hh, ww] = tile_copy;
            else
                # This tile does not contain the source.    Replace the tile with a
                # pseudo-tile that does not have any data in it.
                # The problem is with mp.tile_sources, which can't be allowed to
                # say that an empty tile has a source.
                # TODO: Make a TiledBlob simply an array of an array of tiles
                # rather than a 2d array to avoid this hack.
                empty_tile = ImageTile(b, tile.h_range, tile.w_range,
                                       tile.h_width, tile.w_width,
                                       Array(Float64, 0, 0), tile.constant_background,
                                       tile.epsilon, Array(Float64, 0, 0), tile.iota,
                                       Array(Float64, 0))

                trimmed_tiled_blob[b][hh, ww] = empty_tile;
            end
        end
    end
    Logging.info("Done trimming.")

    trimmed_tiled_blob
end

end
