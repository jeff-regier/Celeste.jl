"""
I/O routines for Dark Energy Camera (DECam) images from the Dark Energy
Camera Legacy Survey (DECaLS).
"""
module DECALSIO

import FITSIO
import ..Celeste: SurveyDataSet, BoundingBox, load_images


struct DECALSDataSet <: SurveyDataSet
basedir::String
"FITS file with CCD metadata, relative to base dir. E.g., survey-ccds-dr5.kd.fits"
metadatafile::String
end

function _linear_pix_to_world(crpix1, crpix2, crval1, crval2,
                              cd1_1, cd1_2, cd2_1, cd2_2, x, y)
    dx = x .- crpix1
    dy = y .- crpix2
    ra = crval1 + cd1_1 .* dx + cd1_2 .* dy
    dec = crval2 + cd2_1 .* dx + cd2_2 .* dy
    return ra, dec
end
    
function _get_overlapping_ccds(dataset::DECALSDataSet, box::BoundingBox)
    f = FITSIO.FITS(joinpath(dataset.basedir, dataset.metadatafile))
    hdu = f[2]::FITSIO.TableHDU
    crpix1 = read(hdu, "crpix1")::Vector{Float32}
    crpix2 = read(hdu, "crpix2")::Vector{Float32}
    crval1 = read(hdu, "crval1")::Vector{Float64}
    crval2 = read(hdu, "crval2")::Vector{Float64}
    cd1_1 = read(hdu, "cd1_1")::Vector{Float32}
    cd1_2 = read(hdu, "cd1_2")::Vector{Float32}
    cd2_1 = read(hdu, "cd2_1")::Vector{Float32}
    cd2_2 = read(hdu, "cd2_2")::Vector{Float32}
    width = read(hdu, "width")::Vector{Int16}
    height = read(hdu, "height")::Vector{Int16}
    close(f)

    function _linear_pix_to_world(x, y)
        dx = x .- crpix1
        dy = y .- crpix2
        ra = crval1 .+ cd1_1 .* dx .+ cd1_2 .* dy
        dec = crval2 .+ cd2_1 .* dx .+ cd2_2 .* dy
        return ra, dec
    end

    # calculate RA, Dec of all CCD corners using a linear WCS approximation
    ra1, dec1 = _linear_pix_to_world(1.0, 1.0)
    ra2, dec2 = _linear_pix_to_world(width, 1.0)
    ra3, dec3 = _linear_pix_to_world(1.0, height)
    ra4, dec4 = _linear_pix_to_world(width, height)

    # Calculate potential overlap by checking the minimum and maximum
    # extent of each CCD in RA/Dec
    # overlap the target bounding box.
    # 
    # It would be better to do this by checking whether each line
    # segment of the CCD boundary intersects any line segment of the
    # bounding box. But this is generally good enough.
    ramin = min.(ra1, ra2, ra3, ra4)
    ramax = max.(ra1, ra2, ra3, ra4)
    decmin = min.(dec1, dec2, dec3, dec4)
    decmax = max.(dec1, dec2, dec3, dec4)

    # We need to treat the discontinuity at RA = 0. We will use a
    # "good-enough", but not complete, solution. In particular, it doesn't
    # account for singularities at the poles and it doesn't treat
    # extremely large (~180 degrees) boxes.  A more complete treatment
    # would use spherical geometry. See
    # https://github.com/spacetelescope/spherical_geometry
    # for an example implementation in Python.
    #
    # Because the corner RA/Dec values (ra1, ra2,...) were derived
    # from crval, crpix, we might have situations where ramin=-0.1,
    # ramax=0.1, or ramin=359.9, ramax=360.1, but fortunately we
    # *don't* need to worry about situations where ramin=0.1,
    # ramax=359.9. (erroneously swapping the east and west limits of
    # the CCD).
    #
    # We also know that the target box doesn't overlap the
    # discontinuity because we required that ramax > ramin when
    # constructing the box.
    #
    # Our strategy is simply to rotate everything away from the
    # discontinuity as far as possible. We find the angle that will
    # center the box at ra = 180, and then rotate all our CCD
    # ramin/ramax by the same angle.
    box_mean_ra = (box.ramax + box.ramin) / 2.0
    offset = 180.0 - box_mean_ra
    box = BoundingBox(box.ramin + offset, box.ramax + offset,
                      box.decmin, box.decmax)
    ramin .= (ramin .+ offset) .% 360.0
    ramax .= (ramax .+ offset) .% 360.0

    # The comparisons are a little unintuitive, because we're looking
    # for *any* overlap.
    mask = ((ramax .> box.ramin) .&
            (ramin .< box.ramax) .&
            (decmax .> box.decmin) .&
            (decmin .< box.decmax))

    # we could return an array of file names and an array of hdu's
    # (from `image_filename` and `image_hdu` columns), but for now just return
    # indices)
    return find(mask)
end


function load_images(dataset::DECALSDataSet, box::BoundingBox)

    # This could return filenames and hdus in which to find images overlapping
    # the box:
    idx = _get_overlapping_ccds(dataset, box)

    # Rest of this function not implemented yet. In order to finish loading
    # Celeste.Image objects from the DECals data, we need to:
    #
    # - Create a function to read in both raw and calibrated images and
    #   determine calibration array (simply divide the calibrated image
    #   by the raw image)
    #
    # - Change the Image type to accept a 2-d rather than 1-d calibration
    #   array (in general, calibration will not be constant in one
    #   dimension like in SDSS.)
    # 
    # - Add `has_background` to Image (make background optional in Image)
    #   Then after calling `load_images(dataset, box)`, check `has_background`
    #   and if false, calculate the background with SEP. This allows us to
    #   optionally read the background from the dataset (e.g., SDSS supplies
    #   one, but DECaLS does not).
    #
    # - Add option to SDSSDataSet to read background or not (e.g., use the
    #   SDSS-calculated background or let Celeste determine the background).
    #
    # - Add general config options for background subtraction,
    #   detection parameters (not actually necessary for this function
    #   but nice to have)
    #
    # - Do something for the PSF (needs investigation into DECaLS PSF model).
    #
    # Testing: make smaller versions of some calibrated and raw images
    # images in that ra, dec range (remove ccds from files) and put
    # them at NERSC for download with a Makefile (Are the raw and
    # calibrated images actually on disk at NERSC?)

    error("Not yet implemented")
end
    


end
