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

