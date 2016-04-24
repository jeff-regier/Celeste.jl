
"""An image, taken though a particular filter band"""
type Image
    # The image height.
    H::Int

    # The image width.
    W::Int

    # An HxW matrix of pixel intensities.
    pixels::Matrix{Float64}

    # The band id (takes on values from 1 to 5).
    b::Int

    # World coordinates
    wcs::WCSTransform

    # The background noise in nanomaggies.
    epsilon::Float64

    # The expected number of photons contributed to this image
    # by a source 1 nanomaggie in brightness.
    iota::Float64
  
    # The components of the point spread function.
    psf::Vector{PsfComponent}

    # SDSS-specific identifiers. A field is a particular region of the sky.
    # A Camcol is the output of one camera column as part of a Run.
    run_num::Int
    camcol_num::Int
    field_num::Int

    # # Field-varying parameters.
    constant_background::Bool
    epsilon_mat::Array{Float64, 2}
    iota_vec::Array{Float64, 1}

    # storing a RawPSF here isn't ideal, because it's an SDSS type
    # not a Celeste type
    raw_psf_comp::RawPSF
end


"""A vector of images, one for each filter band"""
typealias Blob Vector{Image}

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
- h_width: The width of the tile in the h direction
- w_width: The width of the tile in the w direction
- pixels: The pixel values
- remainder: the same as in the Image type.
"""
immutable ImageTile
    b::Int

    h_range::UnitRange{Int}
    w_range::UnitRange{Int}
    h_width::Int
    w_width::Int
    pixels::Matrix{Float64}

    constant_background::Bool
    epsilon::Float64
    epsilon_mat::Matrix{Float64}
    iota::Float64
    iota_vec::Vector{Float64}
end


"""
Return the range of image pixels in an ImageTile.

Args:
  - hh: The tile row index (in 1:number of tile rows)
  - ww: The tile column index (in 1:number of tile columns)
  - H: The number of pixel rows in the image
  - W: The number of pixel columns in the image
  - tile_width: The width and height of a tile in pixels
"""
function tile_range(hh::Int, ww::Int, H::Int, W::Int, tile_width::Int)
    h1 = 1 + (hh - 1) * tile_width
    h2 = min(hh * tile_width, H)
    w1 = 1 + (ww - 1) * tile_width
    w2 = min(ww * tile_width, W)
    h1:h2, w1:w2
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
    h_range, w_range = tile_range(hh, ww, img.H, img.W, tile_width)
    ImageTile(img, h_range, w_range; hh=hh, ww=ww)
end

"""
Constructs an image tile from specific image pixels.

Args:
    - img: The Image to be broken into tiles
    - h_range: A UnitRange for the h pixels
    - w_range: A UnitRange for the w pixels
    - hh: Optional h index in tile coordinates
    - ww: Optional w index in tile coordinates
"""
function ImageTile(img::Image,
                   h_range::UnitRange{Int},
                   w_range::UnitRange{Int};
                   hh::Int=1,
                   ww::Int=1)
    b = img.b
    h_width = maximum(h_range) - minimum(h_range) + 1
    w_width = maximum(w_range) - minimum(w_range) + 1
    pixels = img.pixels[h_range, w_range]

    if img.constant_background
        epsilon_mat = img.epsilon_mat
        iota_vec = img.iota_vec
    else
        # TODO: this subsetting doesn't seem to be working.
        epsilon_mat = img.epsilon_mat[h_range, w_range]
        iota_vec = img.iota_vec[h_range]
    end

    ImageTile(b, h_range, w_range, h_width, w_width, pixels,
                        img.constant_background, img.epsilon, epsilon_mat,
                        img.iota, iota_vec)
end


typealias TiledImage Array{ImageTile, 2}
typealias TiledBlob Vector{TiledImage}

"""
Convert an image to an array of tiles of a given width.

Args:
    - img: An image to be broken into tiles
    - tile_width: The size in pixels of each tile

Returns:
    An array of tiles containing the image.
"""
function break_image_into_tiles(img::Image, tile_width::Int)
    WW = ceil(Int, img.W / tile_width)
    HH = ceil(Int, img.H / tile_width)
    ImageTile[ImageTile(hh, ww, img, tile_width) for hh=1:HH, ww=1:WW]
end


"""
Break a blob into tiles.
"""
function break_blob_into_tiles(blob::Blob, tile_width::Int)
    [break_image_into_tiles(img, tile_width) for img in blob]
end

