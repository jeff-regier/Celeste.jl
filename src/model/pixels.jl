export ActivePixel


# TODO: the identification of active pixels should go in pre-processing
type ActivePixel
    # image index
    n::Int

    # Linear tile index:
    tile_ind::Int

    # Location in tile:
    h::Int
    w::Int
end


"""
Get the active pixels (pixels for which the active sources are present).
TODO: move this to pre-processing and use it instead of setting low-signal
pixels to NaN.
"""
function get_active_pixels(
                      N::Int64,
                      images::Vector{TiledImage},
                      tile_source_map::Vector{Matrix{Vector{Int}}},
                      active_sources::Vector{Int})
    active_pixels = ActivePixel[]

    for n in 1:N, tile_ind in 1:length(images[n].tiles)
        tile_sources = tile_source_map[n][tile_ind]
        if length(intersect(tile_sources, active_sources)) > 0
            tile = images[n].tiles[tile_ind]
            h_width, w_width = size(tile.pixels)
            for w in 1:w_width, h in 1:h_width
                if !Base.isnan(tile.pixels[h, w])
                    push!(active_pixels, ActivePixel(n, tile_ind, h, w))
                end
            end
        end
    end

    active_pixels
end
