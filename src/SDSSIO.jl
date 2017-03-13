# Functions for loading FITS files from the SloanDigitalSkySurvey.
module SDSSIO

using Compat

import FITSIO
import WCS

import ..Log
import ..Model: RawPSF, Image, CatalogEntry, eval_psf
import ..PSF
import Base.convert, Base.getindex


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


function RunCamcolField(run::String, camcol::String, field::String)
    RunCamcolField(
        parse(Int64, run),
        parse(Int64, camcol),
        parse(Int64, field))
end


immutable SkyIntensity
    sky_small::Matrix{Float32}
    sky_x::Vector{Float32}
    sky_y::Vector{Float32}
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

    (xw0 * yw0 * sky.sky_small[x0, y0] + xw1 * yw0 * sky.sky_small[x1, y0] +
         xw0 * yw1 * sky.sky_small[x0, y1] + xw1 * yw1 * sky.sky_small[x1, y1])
end

function getindex(sky::SkyIntensity, i::Int, j::Int)
    interp_sky_kernel(sky, i, j)
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

    # sky intensity has to be strictly greater than 0 for the elbo
    # to be defined
    @assert all((x)-> x > 1e-12, sky_small)

    # interpolate to full sky image
    return SkyIntensity(sky_small, sky_x, sky_y)
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


immutable RawImage
    rcf::RunCamcolField
    b::Int  # band index
    pixels::Matrix{Float32}
    calibration::Vector{Float32}
    sky::SkyIntensity
    wcs::WCS.WCSTransform
    gain::Float32
    raw_psf_comp::RawPSF
end


"""
Read a SDSS run/camcol/field into a vector of RawImages
"""
function load_raw_images(rcf::RunCamcolField, datadir::String)
    subdir2 = "$datadir/$(rcf.run)/$(rcf.camcol)"
    subdir3 = "$subdir2/$(rcf.field)"

    # read gain for each band
    photofield_name = @sprintf("%s/photoField-%06d-%d.fits",
                               subdir2, rcf.run, rcf.camcol)
    gains = read_field_gains(photofield_name, rcf.field)

    # open FITS file containing PSF for each band
    psf_name = @sprintf("%s/psField-%06d-%d-%04d.fit",
                        subdir3, rcf.run, rcf.camcol, rcf.field)
    psffile = FITSIO.FITS(psf_name)

    raw_images = Vector{RawImage}(5)

    for (b, band) in enumerate(['u', 'g', 'r', 'i', 'z'])
        # load image data
        frame_name = @sprintf("%s/frame-%s-%06d-%d-%04d.fits",
                              subdir3, band, rcf.run, rcf.camcol, rcf.field)
        pixels, calibration, sky, wcs = read_frame(frame_name)

        # read mask
        mask_name = @sprintf("%s/fpM-%06d-%s%d-%04d.fit",
                     subdir3, rcf.run, band, rcf.camcol, rcf.field)
        mask_xranges, mask_yranges = read_mask(mask_name)

        # apply mask
        for i=1:length(mask_xranges)
            pixels[mask_xranges[i], mask_yranges[i]] = NaN
        end

        # read the psf
        raw_psf_comp = read_psf(psffile, band)

        # Set it to use a constant background but include the non-constant data.
        raw_images[b] = RawImage(rcf, b,
                            pixels, calibration, sky, wcs,
                            gains[band], raw_psf_comp)
    end

    return raw_images
end


"""
Load all the images for multiple rcfs
"""
function load_raw_images(rcfs::Vector{RunCamcolField}, datadir::String)
    raw_images = RawImage[]

    for rcf in rcfs
        Log.info("loading images for $rcf")
        rcf_raw_images = SDSSIO.load_raw_images(rcf, datadir)
        append!(raw_images, rcf_raw_images)
    end

    raw_images
end


"""
Converts a raw image to an image by fitting the PSF, a mixture of Gaussians.
"""
function convert(::Type{Image}, r::RawImage)
    H, W = size(r.pixels)

    psfstamp = eval_psf(r.raw_psf_comp, H / 2., W / 2.)
    celeste_psf = PSF.fit_raw_psf_for_celeste(psfstamp, 2)[1]

    # scale pixels to raw electron counts
    @assert size(r.pixels, 1) == length(r.calibration)
    epsilon_mat = similar(r.pixels)
    @inbounds for j=1:size(r.pixels, 2), i=1:size(r.pixels, 1)
        sky_ij = interp_sky_kernel(r.sky, i, j)
        r.pixels[i, j] = r.gain * (r.pixels[i, j] / r.calibration[i] + sky_ij)
        epsilon_mat[i, j] = sky_ij * r.calibration[i]
    end

    iota_vec = r.gain ./ r.calibration

    Image(H, W, r.pixels, r.b, r.wcs, celeste_psf,
          r.rcf.run, r.rcf.camcol, r.rcf.field,
          epsilon_mat, iota_vec, r.raw_psf_comp)
end


"""
Read a SDSS run/camcol/field into an array of Images.
"""
function load_field_images(rcfs, stagedir)
    raw_images = SDSSIO.load_raw_images(rcfs, stagedir)

    N = length(raw_images)
    images = Vector{Image}(N)

    Threads.@threads for n in 1:N
        try
            images[n] = convert(Image, raw_images[n])
        catch exc
            Log.exception(exc)
            rethrow()
        end
    end

    @show Base.summarysize(raw_images)
    @show Base.summarysize(images)
    return images
end


# -----------------------------------------------------------------------------
# PSF-related functions

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
    nrow_b = Int((read(hdu, "nrow_b")::Vector{Int32})[1])
    ncol_b = Int((read(hdu, "ncol_b")::Vector{Int32})[1])
    rnrow = (read(hdu, "rnrow")::Vector{Int32})[1]
    rncol = (read(hdu, "rncol")::Vector{Int32})[1]
    cmat_raw = read(hdu, "c")::Array{Float32, 3}
    rrows_raw = read(hdu, "rrows")::Array{Array{Float32,1},1}

    # Only the first (nrow_b, ncol_b) submatrix of cmat is used for
    # reasons obscure to the author.
    cmat = Array{Float64,3}(nrow_b, ncol_b, size(cmat_raw, 3))
    for k=1:size(cmat_raw, 3), j=1:nrow_b, i=1:ncol_b
        cmat[i, j, k] = cmat_raw[i, j, k]
    end

    # convert rrows to Array{Float64, 2}, assuming each row is the same length.
    rrows = Matrix{Float64}(length(rrows_raw[1]), length(rrows_raw))
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
    b = BAND_CHAR_TO_NUM[band]

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
    is_bright    = @compat(read(hdu, "objc_flags")::Vector{Int32} .& Int32(2)) .!= 0
    is_saturated = @compat(read(hdu, "objc_flags")::Vector{Int32} .& Int32(18)) .!= 0
    is_large     = @compat(read(hdu, "objc_flags")::Vector{Int32} .& Int32(24)) .!= 0

    has_child = read(hdu, "nchild")::Vector{Int16} .> 0

    # 6 = star, 3 = galaxy, others can be ignored.
    objc_type = read(hdu, "objc_type")::Vector{Int32}
    is_star = objc_type .== 6
    is_gal = objc_type .== 3
    is_bad_obj = @compat((!).(is_star .| is_gal))

    fracdev = vec((read(hdu, "fracdev")::Matrix{Float32})[b, :])

    # determine mask for rows
    has_dev = fracdev .> 0.
    has_exp = fracdev .< 1.
    is_comp = @compat(has_dev .& has_exp)
    is_bad_fracdev = @compat((fracdev .< 0.) .| (fracdev .> 1))

    # TODO: We don't really want to exclude objects entirely just for being
    # bright: we just don't want to use for scoring (since
    # they're very saturated, presumably).
    mask = @compat((!).(is_bad_fracdev .| is_bad_obj .| is_bright .| has_child))

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
                   "ab_exp"=>vec(ab_exp[b, mask]),
                   "theta_exp"=>vec(theta_exp[b, mask]),
                   "phi_exp"=>vec(phi_exp_deg[b, mask]),
                   "ab_dev"=>vec(ab_dev[b, mask]),
                   "theta_dev"=>vec(theta_dev[b, mask]),
                   "phi_dev"=>vec(phi_dev_deg[b, mask]),
                   "phi_offset"=>vec(phi_offset[b, mask]),
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
    out = Vector{CatalogEntry}(length(catalog["objid"]))

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
    rawcatalogs = Vector{Dict}(length(fts))
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
            mask = @compat(mask .& (cat["mode"] .== 0x01))
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


# TODO: this is obsolete, remove it.
"""
read_photoobj_celeste(fname)

Read a SDSS \"photoobj\" FITS catalog into a Vector{CatalogEntry}.
"""
function read_photoobj_celeste(fname)
    rawcatalog = read_photoobj(fname)
    convert(Vector{CatalogEntry}, rawcatalog)
end

end  # module
