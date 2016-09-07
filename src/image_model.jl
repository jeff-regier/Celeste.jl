"""An image, taken though a particular filter band"""
type Image
    # The image height.
    H::Int

    # The image width.
    W::Int

    # An HxW matrix of pixel intensities.
    pixels::Matrix{Float32}

    # The band id (takes on values from 1 to 5).
    b::Int

    # World coordinates
    wcs::WCSTransform

    # The components of the point spread function.
    psf::Vector{PsfComponent}

    # SDSS-specific identifiers. A field is a particular region of the sky.
    # A Camcol is the output of one camera column as part of a Run.
    run_num::Int
    camcol_num::Int
    field_num::Int

    # The background noise in nanomaggies. (varies by position)
    epsilon_mat::Array{Float32, 2}

    # The expected number of photons contributed to this image
    # by a source 1 nanomaggie in brightness. (varies by row)
    iota_vec::Array{Float32, 1}

    # storing a RawPSF here isn't ideal, because it's an SDSS type
    # not a Celeste type
    raw_psf_comp::RawPSF
end


"""
Tiles of pixels that share the same set of
relevant sources (or other calculations).  It contains all the information
necessary to compute the ELBO and derivatives in this patch of sky.

Note that this cannot be of type Image for the purposes of Celeste
because the raw wcs object is a C++ pointer which julia resonably
refuses to parallelize.
    
Attributes:
- h_range: The h pixel locations of the tile in the original image
- w_range: The w pixel locations of the tile in the original image
- pixels: The pixel values
- remainder: the same as in the Image type.
"""
immutable ImageTile
    b::Int

    h_range::UnitRange{Int}
    w_range::UnitRange{Int}
    pixels::Matrix{Float32}

    epsilon_mat::Matrix{Float32}
    iota_vec::Vector{Float32}
end


"""
Constructs an image tile from an image.

Args:
    - img: The Image to be broken into tiles
    - hh: The tile row index (in 1:number of tile rows)
    - ww: The tile column index (in 1:number of tile columns)
    - tile_width: The width and height of a tile in pixels
"""
function ImageTile(hh::Int, ww::Int, img::Image, tile_width::Int)
    h1 = 1 + (hh - 1) * tile_width
    h2 = min(hh * tile_width, img.H)
    h_range = h1:h2

    w1 = 1 + (ww - 1) * tile_width
    w2 = min(ww * tile_width, img.W)
    w_range = w1:w2

    pixels = img.pixels[h_range, w_range]
    epsilon_mat = img.epsilon_mat[h_range, w_range]
    iota_vec = img.iota_vec[h_range]

    @assert 1 <= img.b <= B
    ImageTile(img.b,
              h_range, w_range,
              pixels, epsilon_mat, iota_vec)
end


"""An image, taken though a particular filter band"""
type TiledImage
    # The image height.
    H::Int

    # The image width.
    W::Int

    # subimages
    tiles::Matrix{ImageTile}

    # all tiles have the same height and width
    tile_width::Int

    # The band id (takes on values from 1 to 5).
    b::Int

    # World coordinates
    wcs::WCSTransform

    # The components of the point spread function.
    psf::Vector{PsfComponent}

    # SDSS-specific identifiers. A field is a particular region of the sky.
    # A Camcol is the output of one camera column as part of a Run.
    run_num::Int
    camcol_num::Int
    field_num::Int

    # storing a RawPSF here isn't ideal, because it's an SDSS type
    # not a Celeste type
    raw_psf_comp::RawPSF
end


function TiledImage(img::Image; tile_width=20)
    WW = ceil(Int, img.W / tile_width)
    HH = ceil(Int, img.H / tile_width)
    tiles = ImageTile[ImageTile(hh, ww, img, tile_width) for hh=1:HH, ww=1:WW]
    TiledImage(img.H, img.W, tiles, tile_width, img.b, img.wcs, img.psf,
               img.run_num, img.camcol_num, img.field_num,
               img.raw_psf_comp)
end


""" Returns the tile containing (or nearest to) the specified pixel"""
function get_containing_tile(pixel_crds::Vector{Float64}, img::TiledImage)
    ht0 = ceil(Int, pixel_crds[1] / img.tile_width)
    wt0 = ceil(Int, pixel_crds[2] / img.tile_width)
    ht = max(1, min(size(img.tiles, 1), ht0))
    wt = max(1, min(size(img.tiles, 2), wt0))
    img.tiles[ht, wt]
end


immutable FlatTiledImage
    # The image height.
    H::Int

    # The image width.
    W::Int

    # subimages
    # TODO: make this a fixed size 2D array
    tiles::Matrix{ImageTile}

    # all tiles have the same height and width
    tile_width::Int

    # The band id (takes on values from 1 to 5).
    b::Int

    # World coordinates
    # TODO: make this a fixed length string/array
    wcs_header::String

    # The components of the point spread function.
    # TODO: make this a fixed size 2D array
    psf::Vector{PsfComponent}

    # SDSS-specific identifiers. A field is a particular region of the sky.
    # A Camcol is the output of one camera column as part of a Run.
    run_num::Int
    camcol_num::Int
    field_num::Int

    # storing a RawPSF here isn't ideal, because it's an SDSS type
    # not a Celeste type
    raw_psf_comp::RawPSF
end


function FlatTiledImage(img::TiledImage)
    wcs_header = WCS.to_header(img.wcs)
    # Kiran, I think the wcs_header will always be shorter than 
    # 10000 characters
    @assert(length(wcs_header) < 10_000)
    FlatTiledImage(img.H, img.W, img.tiles, img.tile_width, img.b, wcs_header,
                   img.psf, img.run_num, img.camcol_num, img.field_num,
                   img.raw_psf_comp)
end


function TiledImage(img::FlatTiledImage)
    wcs_array = WCS.from_header(img.wcs_header)
    @assert(length(wcs_array) == 1)
    wcs = wcs_array[1]
    TiledImage(img.H, img.W, img.tiles, img.tile_width, img.b, wcs,
               img.psf, img.run_num, img.camcol_num, img.field_num,
               img.raw_psf_comp)
end
