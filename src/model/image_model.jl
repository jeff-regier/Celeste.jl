"""
    Image

An image, taken though a particular filter band.
"""
mutable struct Image{B<:AbstractMatrix, P<:AbstractPSFMap}
    H::Int  # image height in pixels
    W::Int  # image width in pixels

    # An HxW matrix of pixel intensities, in raw electron counts (nelec).
    pixels::Matrix{Float32}

    b::Int  # band id (values from 1 to 5).
    wcs::WCSTransform  # Transform between pixels and world coordinates
    psf::Vector{PsfComponent}  # The PSF at the center of the image

    # The background intensity in nanomaggies. (varies by position)
    sky::B

    # The expected number of photons contributed to this image
    # by a source 1 nanomaggie in brightness. (varies by row)
    nelec_per_nmgy::Array{Float32, 1}

    # The image PSF map: a callable f(x, y) -> Matrix that takes a pixel
    # coordinate and returns the rasterized PSF image at that coordinate,
    # with the PSF centered in the image.
    psfmap::P

    Image{B,P}(pixels::Matrix{Float32},
               b::Int,
               wcs::WCSTransform,
               psf::Vector{PsfComponent},
               sky::B,
               nelec_per_nmgy::Array{Float32, 1},
               psfmap::P) where {B<:AbstractMatrix, P<:AbstractPSFMap} =
        new(size(pixels, 1), size(pixels, 2), pixels, b, wcs, psf, sky,
            nelec_per_nmgy, psfmap)
end

Image(pixels::Matrix{Float32},
      b::Int,
      wcs::WCSTransform,
      psf::Vector{PsfComponent},
      sky::B,
      nelec_per_nmgy::Array{Float32, 1},
      psfmap::P) where {B<:AbstractMatrix, P<:AbstractPSFMap} =
    Image{B,P}(pixels, b, wcs, psf, sky, nelec_per_nmgy, psfmap)


# TODO: better name for this.
"""
    calibrated_pixels(im::Image)

Calibrated, sky-subtracted pixel values in nmgy.
"""
calibrated_pixels(im::Image) = im.pixels ./ im.nelec_per_nmgy .- im.sky


# The code below lets us JLD serialize images instances.
# Without this code we get an error for trying to serialize C pointers from WCS
# and some problems for StaticArrays too.
struct SerializedWCS
    header::String
end

JLD.writeas(wcs::WCSTransform) = SerializedWCS(WCS.to_header(wcs))
JLD.readas(s_wcs::SerializedWCS) = WCS.from_header(s_wcs.header)[1]


struct PsfComponentSerial
    alphaBar::Float64
    xiBar::Vector{Float64}
    tauBar::Matrix{Float64}
end

JLD.writeas(pcs::Vector{PsfComponent}) = begin
    [PsfComponentSerial(pc.alphaBar, pc.xiBar, pc.tauBar) for pc in pcs]
end
JLD.readas(pcs::Vector{PsfComponentSerial}) = begin
    [PsfComponent(pc.alphaBar,
                  convert(SVector{2,Float64}, pc.xiBar),
                  convert(SMatrix{2,2,Float64,4}, pc.tauBar)) for pc in pcs]
end
