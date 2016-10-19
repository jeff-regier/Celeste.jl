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

