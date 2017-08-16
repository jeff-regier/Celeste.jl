# See https://github.com/jeff-regier/Celeste.jl/wiki/About-SDSS-and-Stripe-82
struct SkyIntensity
    sky_small::Matrix{Float32} # background flux per pixel, in DNs
    sky_x::Vector{Float32} # interpolation coordinates
    sky_y::Vector{Float32}
    calibration::Vector{Float64} # nMgy per DN for each row
end


"""
Interpolate the 2-d array `sky_small` at the grid of array coordinates spanned
by the vectors `sky_x` and `sky_y` using bilinear interpolation.
The output array will have size `(length(sky_x), length(sky_y))`.
For example, if `x[1] = 3.3` and `y[2] = 4.7`, the element `out[1, 2]`
will be a result of linear interpolation between the values
`sky_small[3:4, 4:5]`.

For coordinates that are out-of-bounds (e.g., `sky_x[i] < 1.0` or
`sky_x[i] > size(sky_small,1)` where the interpolation would index sky_small values
outside the array, the nearest values in the sky_small array are used. (This is
constant extrapolation.)

Return value has units of nMgy.
"""
function interp_sky_kernel(sky::SkyIntensity, i::Int, j::Int)
    nx, ny = size(sky.sky_small)

    y0 = floor(Int, sky.sky_y[j])
    y1 = y0 + 1
    yw0 = sky.sky_y[j] - y0
    yw1 = 1.0f0 - yw0

    # modify out-of-bounds indicies to 1 or ny
    y0 = min(max(y0, 1), ny)
    y1 = min(max(y1, 1), ny)

    x0 = floor(Int, sky.sky_x[i])
    x1 = x0 + 1
    xw0 = sky.sky_x[i] - x0
    xw1 = 1.0f0 - xw0

    # modify out-of-bounds indicies to 1 or nx
    x0 = min(max(x0, 1), nx)
    x1 = min(max(x1, 1), nx)

    # bi-linear interpolation
    sky_dns = (xw0 * yw0 * sky.sky_small[x0, y0]
             + xw1 * yw0 * sky.sky_small[x1, y0]
             + xw0 * yw1 * sky.sky_small[x0, y1]
             + xw1 * yw1 * sky.sky_small[x1, y1])

    # return sky intensity in nMgy
    sky_dns * sky.calibration[i]
end

import Base.getindex

function getindex(sky::SkyIntensity, i::Int, j::Int)
    interp_sky_kernel(sky, i, j)
end


"""An image, taken though a particular filter band"""
mutable struct Image
    # The image height.
    H::Int

    # The image width.
    W::Int

    # An HxW matrix of pixel intensities, in raw electron counts (nelec).
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
    sky::SkyIntensity

    # The expected number of photons contributed to this image
    # by a source 1 nanomaggie in brightness. (varies by row)
    nelec_per_nmgy::Array{Float32, 1}

    # storing a RawPSF here isn't ideal, because it's an SDSS type
    # not a Celeste type
    raw_psf_comp::RawPSF
end
