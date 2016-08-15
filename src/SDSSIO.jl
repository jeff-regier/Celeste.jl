# Functions for loading FITS files from the SloanDigitalSkySurvey.
module SDSSIO

import FITSIO
import WCS

import ..Log
import ..Model: RawPSF, Image, CatalogEntry
import ..PSF
import Base.convert


# types of things to mask in read_mask().
const DEFAULT_MASK_PLANES = ["S_MASK_INTERP",  # bad pixel (was interpolated)
                             "S_MASK_SATUR",  # saturated
                             "S_MASK_CR",  # cosmic ray
                             "S_MASK_GHOST"]  # electronics artifacts

const BAND_CHAR_TO_NUM = Dict('u'=>1, 'g'=>2, 'r'=>3, 'i'=>4, 'z'=>5)


immutable RunCamcolField
    run::Int64
    camcol::Int64
    field::Int64
end


"""
interp_sky(data, xcoords, ycoords)

Interpolate the 2-d array `data` at the grid of array coordinates spanned
by the vectors `xcoords` and `ycoords` using bilinear interpolation.
The output array will have size `(length(xcoords), length(ycoords))`.
For example, if `x[1] = 3.3` and `y[2] = 4.7`, the element `out[1, 2]`
will be a result of linear interpolation between the values
`data[3:4, 4:5]`.

For coordinates that are out-of-bounds (e.g., `xcoords[i] < 1.0` or
`xcoords[i] > size(data,1)` where the interpolation would index data values
outside the array, the nearest values in the data array are used. (This is
constant extrapolation.)
"""
function interp_sky{T, S}(data::Array{T, 2}, xcoords::Vector{S},
                          ycoords::Vector{S})
    nx, ny = size(data)
    out = Array(T, length(xcoords), length(ycoords))
    for j=1:length(ycoords)
        y0 = floor(Int, ycoords[j])
        y1 = y0 + 1
        yw0 = ycoords[j] - y0
        yw1 = one(S) - yw0

        # modify out-of-bounds indicies to 1 or ny
        y0 = min(max(y0, 1), ny)
        y1 = min(max(y1, 1), ny)

        for i=1:length(xcoords)
            x0 = floor(Int, xcoords[i])
            x1 = x0 + 1
            xw0 = xcoords[i] - x0
            xw1 = one(S) - xw0

            # modify out-of-bounds indicies to 1 or nx
            x0 = min(max(x0, 1), nx)
            x1 = min(max(x1, 1), nx)
            @inbounds out[i, j] = (xw0 * yw0 * data[x0, y0] +
                                   xw1 * yw0 * data[x1, y0] +
                                   xw0 * yw1 * data[x0, y1] +
                                   xw1 * yw1 * data[x1, y1])
        end
    end
    return out
end

"""
read_sky(hdu)

Construct an image of the sky from the sky HDU of a SDSS \"frame\" file.
This is typically the 3rd HDU of the file.
"""
function read_sky(hdu::FITSIO.TableHDU)
    sky_small = squeeze(read(hdu, "ALLSKY"), 3)::Array{Float32, 2}
    sky_x = vec(read(hdu, "XINTERP"))::Vector{Float32}
    sky_y = vec(read(hdu, "YINTERP"))::Vector{Float32}

    # convert sky interpolation coordinates from 0-indexed to 1-indexed
    for i=1:length(sky_x)
        sky_x[i] += 1.0f0
    end
    for i=1:length(sky_y)
        sky_y[i] += 1.0f0
    end

    # interpolate to full sky image
    return interp_sky(sky_small, sky_x, sky_y)
end


"""
read_frame(fname)

Read an SDSS \"frame\" FITS file and return a 4-tuple:

- `image`: sky-subtracted and calibrated 2-d image.
- `calibration`: 1-d \"calibration\".
- `sky`: 2-d sky that was subtracted.
- `wcs`: WCSTransform constructed from image header.
"""
function read_frame(fname)
    f = FITSIO.FITS(fname)
    hdr = FITSIO.read_header(f[1], String)::String
    image = read(f[1])::Array{Float32, 2}  # sky-subtracted & calibrated data
    calibration = read(f[2])::Vector{Float32}
    sky = read_sky(f[3])
    close(f)

    wcs = WCS.from_header(hdr)[1]

    return image, calibration, sky, wcs
end

"""
decalibrate!(image, sky, calibration, gain)

Convert `image` to raw counts. `image` is modified in-place.

Note that result may not be close to integers due to the analog to digital
conversion process in the instrument.
"""
function decalibrate!{T<:Number}(image::Array{T, 2}, sky::Array{T, 2},
                                 calibration::Vector{T}, gain::Number)
    @assert size(image) == size(sky)
    @assert size(image, 1) == length(calibration)
    for j=1:size(image, 2), i=1:size(image, 1)
        @inbounds image[i, j] = gain * (image[i, j] / calibration[i] +
                                        sky[i, j])
    end
end


"""
read_field_gains(fname, fieldnum)

Return the image gains for field number `fieldnum` in an SDSS
\"photoField\" file `fname`.
"""
function read_field_gains(fname, fieldnum::Integer)

    f = FITSIO.FITS(fname)
    fieldnums = read(f[2], "FIELD")::Vector{Int32}
    gains = read(f[2], "GAIN")::Array{Float32, 2}
    close(f)

    # Find first occurance of `fieldnum` and return the corresponding gain.
    for i=1:length(fieldnums)
        if fieldnums[i] == fieldnum
            bandgains = gains[:, i]
            return Dict(zip("ugriz", bandgains))
        end
    end

    error("field number $fieldnum not found in file: $fname")
end


"""
read_mask(fname[, mask_planes])

Read a \"fpM\"-format SDSS file and return masked image ranges,
based on `mask_planes`. Returns two `Vector{UnitRange{Int}}`,
giving the range of x and y indicies to be masked.
"""
function read_mask(fname, mask_planes=DEFAULT_MASK_PLANES)
    f = FITSIO.FITS(fname)

    # The last (12th) HDU contains a key describing what each of the
    # other HDUs are. Use this to find the indicies of all the relevant
    # HDUs (those with attributeName matching a value in `mask_planes`).
    value = read(f[12], "Value")::Vector{Int32}
    def = read(f[12], "defName")::Vector{String}
    attribute = read(f[12], "attributeName")::Vector{String}

    # initialize return values
    xranges = UnitRange{Int}[]
    yranges = UnitRange{Int}[]

    # Loop over keys and check if each is a mask plane we're interested in
    # (those with defName == "S_MASKTYPE" and "attributeName" in mask_planes).
    # If so, read from the corresponding HDU and construct ranges to mask.
    for i=1:length(value)
        if (def[i] == "S_MASKTYPE" && attribute[i] in mask_planes)

            # `value` starts from 0, but first table hdu is hdu number 2
            hdunum = value[i] + 2

            cmin = read(f[hdunum], "cmin")::Vector{Int32}
            cmax = read(f[hdunum], "cmax")::Vector{Int32}
            rmin = read(f[hdunum], "rmin")::Vector{Int32}
            rmax = read(f[hdunum], "rmax")::Vector{Int32}

            # "c" ("column") refers to mask's NAXIS1 (x axis);
            # "r" ("row") refers to mask's NAXIS2 (y axis).
            # cmin/cmax and rmin/rmax are 0-based and inclusive, so we add 1
            # to both.
            for j=1:length(cmin)
                push!(xranges, (cmin[j] + 1):(cmax[j] + 1))
                push!(yranges, (rmin[j] + 1):(rmax[j] + 1))
            end
        end
    end

    return xranges, yranges
end


"""
Read a SDSS run/camcol/field into an array of Images. `frame_dir` is the
directory in which to find SDSS \"frame\" files. This function accepts
optional keyword arguments `fpm_dir`, `psfield_dir` and `photofield_dir`
giving the directories of fpM, psField and photoField files. The defaults
for these arguments is `frame_dir`.
"""
function load_field_images(ft::RunCamcolField, datadir::String)
    subdir2 = "$datadir/$(ft.run)/$(ft.camcol)"
    subdir3 = "$subdir2/$(ft.field)"

    # read gain for each band
    photofield_name = @sprintf("%s/photoField-%06d-%d.fits",
                               subdir2, ft.run, ft.camcol)
    gains = read_field_gains(photofield_name, ft.field)

    # open FITS file containing PSF for each band
    psf_name = @sprintf("%s/psField-%06d-%d-%04d.fit",
                        subdir3, ft.run, ft.camcol, ft.field)
    psffile = FITSIO.FITS(psf_name)

    result = Array(Image, 5)

    for (bandnum, band) in enumerate(['u', 'g', 'r', 'i', 'z'])

        # load image data
        frame_name = @sprintf("%s/frame-%s-%06d-%d-%04d.fits",
                              subdir3, band, ft.run, ft.camcol, ft.field)
        data, calibration, sky, wcs = read_frame(frame_name)

        # scale data to raw electron counts
        decalibrate!(data, sky, calibration, gains[band])

        # read mask
        mask_name = @sprintf("%s/fpM-%06d-%s%d-%04d.fit",
                     subdir3, ft.run, band, ft.camcol, ft.field)
        mask_xranges, mask_yranges = read_mask(mask_name)

        # apply mask
        for i=1:length(mask_xranges)
            data[mask_xranges[i], mask_yranges[i]] = NaN
        end

        H, W = size(data)

        # read the psf
        sdsspsf = read_psf(psffile, band)

        # evalute the psf in the center of the image and then fit it.
        psfstamp = sdsspsf(H / 2., W / 2.)
        psf = PSF.fit_raw_psf_for_celeste(psfstamp)[1]

        # For now, use the median noise and sky.  Here,
        # epsilon * iota needs to be in units comparable to nelec
        # electron counts.
        # Note that each are actuall pretty variable.
        iota_vec = convert(Vector{Float64}, gains[band] ./ calibration)
        epsilon_mat = convert(Array{Float64, 2}, sky .* calibration)

        # Set it to use a constant background but include the non-constant data.
        result[bandnum] = Image(H, W, data, bandnum, wcs, psf,
                                ft.run, ft.camcol, ft.field, epsilon_mat,
                                iota_vec, sdsspsf)
    end

    return result
end

# -----------------------------------------------------------------------------
# PSF-related functions

"""
psf(x, y)

Evaluate the PSF at the given image coordinates. The size of the result is
will be `(psf.rnrow, psf.rncol)`, with the PSF (presumably) centered in the
stamp.

This function was originally based on the function sdss_psf_at_points
in astrometry.net:
https://github.com/dstndstn/astrometry.net/blob/master/util/sdss_psf.py
"""
function (psf::RawPSF)(x::Real, y::Real)
    const RCS = 0.001  # A coordinate transform to keep polynomial
                       # coefficients to a reasonable size.
    nk = size(psf.rrows, 2)  # number of eigen images.

    # initialize output stamp
    stamp = zeros(psf.rnrow, psf.rncol)

    # Loop over eigen images
    for k=1:nk
        # calculate the weight for the k-th eigen image from psf.cmat.
        # Note that the image coordinates and coefficients are intended
        # to be zero-indexed.
        w = 0.0
        for j=1:size(psf.cmat, 2), i=1:size(psf.cmat, 1)
            w += (psf.cmat[i, j, k] *
                  (RCS * (x - 1.0))^(i-1) * (RCS * (y - 1.0))^(j-1))
        end

        # add the weighted k-th eigen image to the output stamp
        for i=1:length(stamp)
            stamp[i] += w * psf.rrows[i, k]
        end
    end

    return stamp
end


"""
read_psf(fitsfile, band)

Read PSF components for the given band ('u', 'g', 'r', 'i', or 'z')
from the open FITSIO.FITS instance `fitsfile`, which should be a SDSS
\"psField\" file, and return a RawPSF instance representing the spatially
variable PSF.
"""
function read_psf(fitsfile::FITSIO.FITS, band::Char)

    # get the HDU, assuming PSF hdus are in order after primary header
    extnum = 1 + BAND_CHAR_TO_NUM[band]
    hdu = fitsfile[extnum]::FITSIO.TableHDU

    nrows = FITSIO.read_key(hdu, "NAXIS2")[1]::Int
    nrow_b = (read(hdu, "nrow_b")::Vector{Int32})[1]
    ncol_b = (read(hdu, "ncol_b")::Vector{Int32})[1]
    rnrow = (read(hdu, "rnrow")::Vector{Int32})[1]
    rncol = (read(hdu, "rncol")::Vector{Int32})[1]
    cmat_raw = read(hdu, "c")::Array{Float32, 3}
    rrows_raw = read(hdu, "rrows")::Array{Array{Float32,1},1}

    # Only the first (nrow_b, ncol_b) submatrix of cmat is used for
    # reasons obscure to the author.
    cmat = Array(Float64, nrow_b, ncol_b, size(cmat_raw, 3))
    for k=1:size(cmat_raw, 3), j=1:nrow_b, i=1:ncol_b
        cmat[i, j, k] = cmat_raw[i, j, k]
    end

    # convert rrows to Array{Float64, 2}, assuming each row is the same length.
    rrows = Array(Float64, length(rrows_raw[1]), length(rrows_raw))
    for i=1:length(rrows_raw)
        rrows[:, i] = rrows_raw[i]
    end

    return RawPSF(rrows, rnrow, rncol, cmat)
end


"""
Read a source catalog from an SDSS \"photoObj\" FITS file for a given
run/camcol/field combination, returning a Vector of columns and a Vector
of column names.

This is currently pretty specific to Celeste because it returns only columns
that we're interested in, and it does some transformations of some of the
columns.

The photoObj file format is documented here:
https://data.sdss.org/datamodel/files/BOSS_PHOTOOBJ/RERUN/RUN/CAMCOL/photoObj.html
"""
function read_photoobj(fname, band::Char='r')

    bandnum = BAND_CHAR_TO_NUM[band]

    f = FITSIO.FITS(fname)

    # sometimes the expected table extension is only an empty generic FITS
    # header, indicating no objects. We check explicitly for this case here
    # and return an empty table
    if (length(f) < 2) || (!isa(f[2], FITSIO.TableHDU))
        catalog = Dict("objid"=>String[],
                       "ra"=>Float64[],
                       "dec"=>Float64[],
                       "thing_id"=>Int32[],
                       "mode"=>UInt8[],
                       "is_star"=>BitVector(),
                       "is_gal"=>BitVector(),
                       "frac_dev"=>Float32[],
                       "ab_exp"=>Float32[],
                       "theta_exp"=>Float32[],
                       "phi_exp"=>Float32[],
                       "ab_dev"=>Float32[],
                       "theta_dev"=>Float32[],
                       "phi_dev"=>Float32[],
                       "phi_offset"=>Float32[])
        for (b, n) in BAND_CHAR_TO_NUM
            catalog["psfflux_$b"] = Float32[]
            catalog["compflux_$b"] = Float32[]
            catalog["expflux_$b"] = Float32[]
            catalog["devflux_$b"] = Float32[]
        end
        return catalog
    end

    hdu = f[2]::FITSIO.TableHDU

    objid = read(hdu, "objid")::Vector{String}
    ra = read(hdu, "ra")::Vector{Float64}
    dec = read(hdu, "dec")::Vector{Float64}
    mode = read(hdu, "mode")::Vector{UInt8}
    thing_id = read(hdu, "thing_id")::Vector{Int32}

    # Get "bright" objects.
    # (In objc_flags, the bit pattern Int32(2) corresponds to bright objects.)
    is_bright = read(hdu, "objc_flags")::Vector{Int32} & Int32(2) .!= 0
    is_saturated = read(hdu, "objc_flags")::Vector{Int32} & Int32(18) .!= 0
    is_large = read(hdu, "objc_flags")::Vector{Int32} & Int32(24) .!= 0

    has_child = read(hdu, "nchild")::Vector{Int16} .> 0

    # 6 = star, 3 = galaxy, others can be ignored.
    objc_type = read(hdu, "objc_type")::Vector{Int32}
    is_star = objc_type .== 6
    is_gal = objc_type .== 3
    is_bad_obj = !(is_star | is_gal)

    fracdev = vec((read(hdu, "fracdev")::Matrix{Float32})[bandnum, :])

    # determine mask for rows
    has_dev = fracdev .> 0.
    has_exp = fracdev .< 1.
    is_comp = has_dev & has_exp
    is_bad_fracdev = (fracdev .< 0.) | (fracdev .> 1)

    # TODO: We don't really want to exclude objects entirely just for being
    # bright: we just don't want to use for scoring (since
    # they're very saturated, presumably).
    mask = !(is_bad_fracdev | is_bad_obj | is_bright | has_child)

    # Read the fluxes.
    # Record the cmodelflux if the galaxy is composite, otherwise use
    # the flux for the appropriate type.
    psfflux = read(hdu, "psfflux")::Matrix{Float32}
    cmodelflux = read(hdu, "cmodelflux")::Matrix{Float32}
    devflux = read(hdu, "devflux")::Matrix{Float32}
    expflux = read(hdu, "expflux")::Matrix{Float32}

    # We actually only store the following properties for the given band
    # (see below in Dict construction) NB: the phi quantites in tractor are
    # multiplied by -1.
    phi_dev_deg = read(hdu, "phi_dev_deg")::Matrix{Float32}
    phi_exp_deg = read(hdu, "phi_exp_deg")::Matrix{Float32}
    phi_offset = read(hdu, "phi_offset")::Matrix{Float32}

    theta_dev = read(hdu, "theta_dev")::Matrix{Float32}
    theta_exp = read(hdu, "theta_exp")::Matrix{Float32}

    ab_exp = read(hdu, "ab_exp")::Matrix{Float32}
    ab_dev = read(hdu, "ab_dev")::Matrix{Float32}

    close(f)

    # construct result catalog
    catalog = Dict("objid"=>objid[mask],
                   "ra"=>ra[mask],
                   "dec"=>dec[mask],
                   "thing_id"=>thing_id[mask],
                   "mode"=>mode[mask],
                   "is_star"=>is_star[mask],
                   "is_gal"=>is_gal[mask],
                   "frac_dev"=>fracdev[mask],
                   "ab_exp"=>vec(ab_exp[bandnum, mask]),
                   "theta_exp"=>vec(theta_exp[bandnum, mask]),
                   "phi_exp"=>vec(phi_exp_deg[bandnum, mask]),
                   "ab_dev"=>vec(ab_dev[bandnum, mask]),
                   "theta_dev"=>vec(theta_dev[bandnum, mask]),
                   "phi_dev"=>vec(phi_dev_deg[bandnum, mask]),
                   "phi_offset"=>vec(phi_offset[bandnum, mask]),
                   "is_large"=>vec(is_large[mask]),
                   "is_saturated"=>vec(is_saturated[mask]))
    for (b, n) in BAND_CHAR_TO_NUM
        catalog["psfflux_$b"] = vec(psfflux[n, mask])
        catalog["compflux_$b"] = vec(cmodelflux[n, mask])
        catalog["expflux_$b"] = vec(expflux[n, mask])
        catalog["devflux_$b"] = vec(devflux[n, mask])
    end

    # left over note about fluxes above: Use the cmodelflux if the
    # galaxy is composite, otherwise use the flux for the appropriate type.

    return catalog
end


"""
Convert from a catalog in dictionary-of-arrays, as returned by
read_photoobj to Vector{CatalogEntry}.
"""
function convert(::Type{Vector{CatalogEntry}}, catalog::Dict{String, Any})
    out = Array(CatalogEntry, length(catalog["objid"]))

    for i=1:length(catalog["objid"])
        worldcoords = [catalog["ra"][i], catalog["dec"][i]]
        frac_dev = catalog["frac_dev"][i]

        # Fill star and galaxy flux
        star_fluxes = zeros(5)
        gal_fluxes = zeros(5)
        for (j, band) in enumerate(['u', 'g', 'r', 'i', 'z'])

            # Make negative fluxes positive.
            # TODO: How can there be negative fluxes?
            psfflux = max(catalog["psfflux_$band"][i], 1e-6)
            devflux = max(catalog["devflux_$band"][i], 1e-6)
            expflux = max(catalog["expflux_$band"][i], 1e-6)

            # galaxy flux is a weighted sum of the two components.
            star_fluxes[j] = psfflux
            gal_fluxes[j] = frac_dev * devflux + (1.0 - frac_dev) * expflux
        end

        # For shape parameters, we use the dominant component (dev or exp)
        usedev = frac_dev > 0.5
        fits_ab = usedev? catalog["ab_dev"][i] : catalog["ab_exp"][i]
        fits_phi = usedev? catalog["phi_dev"][i] : catalog["phi_exp"][i]
        fits_theta = usedev? catalog["theta_dev"][i] : catalog["theta_exp"][i]

        # effective radius
        re_arcsec = max(fits_theta, 1. / 30)
        re_pixel = re_arcsec / 0.396

        fits_phi -= catalog["phi_offset"][i]

        # fits_phi is now degrees counter-clockwise from vertical
        # in pixel coordinates.

        # Celeste's phi measures radians counter-clockwise from vertical.
        celeste_phi_rad = fits_phi * (pi / 180)

        entry = CatalogEntry(worldcoords, catalog["is_star"][i], star_fluxes,
                             gal_fluxes, frac_dev, fits_ab, celeste_phi_rad, re_pixel,
                             catalog["objid"][i], Int(catalog["thing_id"][i]))
        out[i] = entry
    end

    return out
end


"""
read_photoobj_files(fieldids, dirs) -> Vector{CatalogEntry}

Combine photoobj catalogs for the given overlapping fields, returning a single
joined catalog.

The `duplicate_policy` argument controls how catalogs are joined.
With `duplicate_policy = :primary`, only primary objects are included in the
combined catalog.
With `duplicate_policy = :first`, only the first detection is included in the
combined catalog.
"""
function read_photoobj_files(fts::Vector{RunCamcolField},
                             datadir::String;
                             duplicate_policy=:primary)
    @assert duplicate_policy == :primary || duplicate_policy == :first
    @assert duplicate_policy == :primary || length(fts) == 1

    Log.info("reading photoobj catalogs for $(length(fts)) fields")

    # the code below assumes there is at least one field.
    if length(fts) == 0
        return CatalogEntry[]
    end

    # Read in all photoobj catalogs.
    rawcatalogs = Array(Dict, length(fts))
    for i in eachindex(fts)
        ft = fts[i]
        dir = "$datadir/$(ft.run)/$(ft.camcol)/$(ft.field)"
        fname = @sprintf "%s/photoObj-%06d-%d-%04d.fits" dir ft.run ft.camcol ft.field
        Log.info("field $(fts[i]): reading $fname")
        rawcatalogs[i] = read_photoobj(fname)
    end

    for i in eachindex(fts)
        Log.info("field $(fts[i]): $(length(rawcatalogs[i]["objid"])) entries")
    end

    # Limit each catalog to primary objects and objects where thing_id != -1
    # (thing_id == -1 indicates that the matching process failed)
    for cat in rawcatalogs
        mask = (cat["thing_id"] .!= -1)
        if duplicate_policy == :primary
            mask &= (cat["mode"] .== 0x01)
        end
        for key in keys(cat)
            cat[key] = cat[key][mask]
        end
    end

    for i in eachindex(fts)
        Log.info(string("field $(fts[i]): $(length(rawcatalogs[i]["objid"])) ",
                "filtered entries"))
    end

    # Merge all catalogs together (there should be no duplicate objects,
    # because for each object there should only be one "primary" occurance.)
    rawcatalog = deepcopy(rawcatalogs[1])
    for i=2:length(rawcatalogs)
        for key in keys(rawcatalog)
            append!(rawcatalog[key], rawcatalogs[i][key])
        end
    end

    # check that there are no duplicate thing_ids (see above comment)
    if length(Set(rawcatalog["thing_id"])) < length(rawcatalog["thing_id"])
        error("Found one or more duplicate primary thing_ids in photoobj " *
              "catalogs")
    end

    # convert to celeste format catalog
    catalog = convert(Vector{CatalogEntry}, rawcatalog)

    return catalog
end


"""
read_photoobj_celeste(fname)

Read a SDSS \"photoobj\" FITS catalog into a Vector{CatalogEntry}.
"""
function read_photoobj_celeste(fname)
    rawcatalog = read_photoobj(fname)
    convert(Vector{CatalogEntry}, rawcatalog)
end

end  # module
