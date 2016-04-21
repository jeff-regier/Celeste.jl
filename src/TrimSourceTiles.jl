"""
This module contains code that couldn't go in Model because of
its dependence on ElboDeriv. (ElboDeriv imports Model
and this code imports ElboDeriv.)
Eventually we won't be using ElboDeriv for initialization,
once we don't have pre-existing catalogs for initialization,
and we have to use a blob extraction routine.
Until then, this code is isolated here. (An alternative would
be to include ElboDeriv in the Model, but I think it's more
clear to keep those separate, at the cost of having to put
trim_source_tiles in it's own module.)
"""

module TrimSourceTiles

using ..Types
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

        pix_loc = WCSUtils.world_to_pix(mp.patches[s, b], mp.vp[s][ids.u]);

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
