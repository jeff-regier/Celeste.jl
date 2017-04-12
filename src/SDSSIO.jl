# Functions for loading FITS files from the SloanDigitalSkySurvey.
module SDSSIO

import FITSIO
import WCS

import ..Log
import ..Model: RawPSF, Image, CatalogEntry, eval_psf, SkyIntensity
import ..PSF
import Base.convert, Base.getindex


# types of things to mask in read_mask().
const DEFAULT_MASK_PLANES = ["S_MASK_INTERP",  # bad pixel (was interpolated)
                             "S_MASK_SATUR",  # saturated
                             "S_MASK_CR",  # cosmic ray
                             "S_MASK_GHOST"]  # electronics artifacts

const BAND_CHAR_TO_NUM = Dict('u'=>1, 'g'=>2, 'r'=>3, 'i'=>4, 'z'=>5)


immutable RunCamcolField
    run::Int16
    camcol::UInt8
    field::Int16
end


function RunCamcolField(run::String, camcol::String, field::String)
    RunCamcolField(
        parse(Int16, run),
        parse(UInt8, camcol),
        parse(Int16, field))
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

    return sky_small, sky_x, sky_y
end

function slurp_fits(fname)
    is_linux() || error("Slurping not implemented on this OS")
    # Do this with explicit POSIX calls, to make sure to avoid
    # any well-intended, but ultimately unhelpful intermediate
    # buffering.
    fd = ccall(:open, Cint, (Ptr{UInt8}, Cint, Cint), fname, Base.Filesystem.JL_O_RDONLY, 0)
    systemerror("open", fd == -1)
    stat_struct = zeros(UInt8, ccall(:jl_sizeof_stat, Int32, ()))
    # Can't use Base.stat because threading
    ret = ccall(:jl_fstat, Cint, (Cint, Ptr{UInt8}), fd, stat_struct)
    ret == 0 || throw(Base.UVError("stat",r))
    size = Base.Filesystem.StatStruct(stat_struct).size
    data = Array(UInt8, size)
    rsize = ccall(:read, Cint, (Cint, Ptr{UInt8}, Csize_t), fd, data, size)
    systemerror("read", rsize != size)
    r = ccall(:close, Cint, (Cint,), fd)
    systemerror("close", r == -1)
    data
end

"""
read_frame(fname)

Read an SDSS \"frame\" FITS file and return a 4-tuple:

- `image`: sky-subtracted and calibrated 2-d image.
- `calibration`: 1-d \"calibration\".
- `sky`: 2-d sky that was subtracted.
- `wcs`: WCSTransform constructed from image header.
"""
function read_frame(fname; data=nothing)
    f = FITSIO.FITS(data != nothing ? data : fname)
    hdr = FITSIO.read_header(f[1], String)::String
    image = read(f[1])::Array{Float32, 2}  # sky-subtracted & calibrated data
    calibration = read(f[2])::Vector{Float32}
    sky_small, sky_x, sky_y = read_sky(f[3])
    close(f)

    sky = SkyIntensity(sky_small, sky_x, sky_y, calibration)

    wcs = WCS.from_header(hdr)[1]

    return image, calibration, sky, wcs
end


"""
read_field_gains(fname, fieldnum)

Return the image gains for field number `fieldnum` in an SDSS
\"photoField\" file `fname`.
"""
function read_field_gains(fname, fieldnum::Integer; data=nothing)
    f = FITSIO.FITS(data != nothing ? data : fname)
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
function read_mask(fname, mask_planes=DEFAULT_MASK_PLANES; data=nothing)
    f = FITSIO.FITS(data != nothing ? data : fname)

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
    close(f)

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
function load_raw_images(rcf::RunCamcolField, datadir, slurp::Bool = false, drop_quickly::Bool = false)
    basedir = isa(datadir, String) ? datadir : datadir(rcf)
    subdir2 = "$basedir/$(rcf.run)/$(rcf.camcol)"
    subdir3 = "$subdir2/$(rcf.field)"

    # read gain for each band
    photofield_name = "$subdir2/photoField-$(dec(rcf.run,6))-$(rcf.camcol).fits"
    gains = read_field_gains(photofield_name, rcf.field;
                data=(slurp ? slurp_fits(photofield_name) : nothing))

    # open FITS file containing PSF for each band
    psf_name = "$(subdir3)/psField-$(dec(rcf.run,6))-$(rcf.camcol)-$(dec(rcf.field,4)).fit"
    psffile = FITSIO.FITS(slurp ? slurp_fits(psf_name) : psf_name)

    raw_images = Vector{RawImage}(5)

    for (b, band) in enumerate(['u', 'g', 'r', 'i', 'z'])
        # load image data
        frame_name = "$(subdir3)/frame-$(band)-$(dec(rcf.run,6))-$(rcf.camcol)-$(dec(rcf.field,4)).fits"
        mask_name = "$(subdir3)/fpM-$(dec(rcf.run,6))-$(band)$(rcf.camcol)-$(dec(rcf.field,4)).fit"
        r = parse_image(rcf, b, frame_name, mask_name, psffile, band, gains)
        if !drop_quickly
            raw_images[b] = r
        end
    end
    close(psffile)

    return raw_images
end

function parse_image(rcf, b, frame_name, mask_name, psffile, band, gains; frame_data=nothing, mask_data=nothing)
    @assert !isempty(frame_name) || frame_data != nothing
    @assert !isempty(mask_name) || mask_data != nothing

    # load image data
    pixels, calibration, sky, wcs = read_frame(frame_name; data=frame_data)

    # read mask
    mask_xranges, mask_yranges = read_mask(mask_name; data=mask_data)

    # apply mask
    for i=1:length(mask_xranges)
        pixels[mask_xranges[i], mask_yranges[i]] = NaN
    end

    # read the psf
    raw_psf_comp = read_psf(psffile, band)

    # Set it to use a constant background but include the non-constant data.
    r = RawImage(rcf, b,
                        pixels, calibration, sky, wcs,
                        gains[band], raw_psf_comp)
    r
end


"""
Load all the images for multiple rcfs
"""
function load_raw_images(rcfs::Vector{RunCamcolField}, datadir, slurp::Bool = false, drop_quickly::Bool = false)
    raw_images = RawImage[]

    for rcf in rcfs
        #Log.info("loading images for $rcf")
        rcf_raw_images = SDSSIO.load_raw_images(rcf, datadir, slurp, drop_quickly)
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
    @inbounds for j=1:size(r.pixels, 2), i=1:size(r.pixels, 1)
        sky_ij = r.sky[i, j]
        r.pixels[i, j] = (r.gain / r.calibration[i]) * (r.pixels[i, j] + sky_ij)
    end

    iota_vec = r.gain ./ r.calibration

    Image(H, W, r.pixels, r.b, r.wcs, celeste_psf,
          r.rcf.run, r.rcf.camcol, r.rcf.field,
          r.sky, iota_vec, r.raw_psf_comp)
end


"""
Read a SDSS run/camcol/field into an array of Images.
"""
function load_field_images(rcfs, stagedir, slurp::Bool = false, drop_quickly::Bool = false)
    raw_images = SDSSIO.load_raw_images(rcfs, stagedir, slurp, drop_quickly)

    N = length(raw_images)
    images = Vector{Image}(N)

    drop_quickly && return images

    #Threads.@threads for n in 1:N
    for n in 1:N
        try
            images[n] = convert(Image, raw_images[n])
        catch exc
            Log.exception(exc)
            rethrow()
        end
    end

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
function read_photoobj(fname, band::Char='r'; data=nothing)
    b = BAND_CHAR_TO_NUM[band]

    f = FITSIO.FITS(data != nothing ? data : fname)

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
    objc_flags = read(hdu, "objc_flags")::Vector{Int32}
    objc_flags2 = read(hdu, "objc_flags2")::Vector{Int32}

    # bright, saturated, or large
    bad_flags1 = objc_flags .& UInt32(1^2 + 2^18 + 2^24) .!= 0

    # nopeak, DEBLEND_DEGENERATE, or saturated center
    bad_flags2 = objc_flags2 .& UInt32(2^14 + 2^18 + 2^11) .!= 0
    i = findfirst(objid, "1237680069097291856")

    has_child = read(hdu, "nchild")::Vector{Int16} .> 0

    # 6 = star, 3 = galaxy, others can be ignored.
    objc_type = read(hdu, "objc_type")::Vector{Int32}
    is_star = objc_type .== 6
    is_gal = objc_type .== 3
    bad_type = ((!).(is_star .| is_gal))

    fracdev = vec((read(hdu, "fracdev")::Matrix{Float32})[b, :])

    # determine mask for rows
    has_dev = fracdev .> 0.
    has_exp = fracdev .< 1.
    is_comp = (has_dev .& has_exp)
    is_bad_fracdev = ((fracdev .< 0.) .| (fracdev .> 1))

    # TODO: We don't really want to exclude objects entirely just for being
    # bright: we just don't want to use for scoring (since
    # they're very saturated, presumably).
    mask = (!).(is_bad_fracdev .| bad_type .| bad_flags1 .| bad_flags2 .| has_child)

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
                   "phi_offset"=>vec(phi_offset[b, mask]))
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
    out = CatalogEntry[]

    for i=1:length(catalog["objid"])
        worldcoords = [catalog["ra"][i], catalog["dec"][i]]
        frac_dev = catalog["frac_dev"][i]

        # Fill star and galaxy flux
        star_fluxes = zeros(5)
        gal_fluxes = zeros(5)
        for (j, band) in enumerate(['u', 'g', 'r', 'i', 'z'])
            # Make negative fluxes positive.
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
        push!(out, entry)
    end

    return out
end


function assemble_catalog(rawcatalogs::Vector{Dict};
                          duplicate_policy=:primary)
    # Limit each catalog to primary objects and objects where thing_id != -1
    # (thing_id == -1 indicates that the matching process failed)
    for cat in rawcatalogs
        mask = (cat["thing_id"] .!= -1)
        if duplicate_policy == :primary
            mask = (mask .& (cat["mode"] .== 0x01))
        end
        for key in keys(cat)
            cat[key] = cat[key][mask]
        end
    end

    #for i in eachindex(fts)
    #    Log.info(string("field $(fts[i]): $(length(rawcatalogs[i]["objid"])) ",
    #            "filtered entries"))
    #end

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
read_photoobj_files(fieldids, dirs) -> Vector{CatalogEntry}

Combine photoobj catalogs for the given overlapping fields, returning a single
joined catalog.

The `duplicate_policy` argument controls how catalogs are joined.
With `duplicate_policy = :primary`, only primary objects are included in the
combined catalog.
With `duplicate_policy = :first`, only the first detection is included in the
combined catalog.
"""
function read_photoobj_files(fts::Vector{RunCamcolField}, datadir;
                             duplicate_policy=:primary,
                             slurp::Bool = false, drop_quickly::Bool = false)
    @assert duplicate_policy == :primary || duplicate_policy == :first
    @assert duplicate_policy == :primary || length(fts) == 1

    #Log.info("reading photoobj catalogs for $(length(fts)) fields")

    # the code below assumes there is at least one field.
    if length(fts) == 0
        return CatalogEntry[]
    end

    # Read in all photoobj catalogs.
    rawcatalogs = Vector{Dict}(length(fts))
    for i in eachindex(fts)
        ft = fts[i]
        basedir = isa(datadir, String) ? datadir : datadir(ft)
        dir = "$basedir/$(ft.run)/$(ft.camcol)/$(ft.field)"
        fname = "$dir/photoObj-$(dec(ft.run,6))-$(dec(ft.camcol))-$(dec(ft.field,4)).fits"
        #Log.info("field $(fts[i]): reading $fname")
        po = read_photoobj(fname, 'r';
                           data=(slurp ? slurp_fits(fname) : nothing))
        if !drop_quickly
            rawcatalogs[i] = po
        end
    end

    if drop_quickly
        return CatalogEntry[]
    end

    #for i in eachindex(fts)
    #    Log.info("field $(fts[i]): $(length(rawcatalogs[i]["objid"])) entries")
    #end

    return assemble_catalog(rawcatalogs; duplicate_policy=duplicate_policy)
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


# -----------------------------------------------------------------------------
# big file I/O

const NumRuns = 761
const NumCamcols = 6
const EachPhotoField = 3*1024*1024

const NumRCFs = 924038
const EachRCF = 97*1024*1024
const GoodTransferSize = 4*1024*1024
const EachBigFile = 3298534883328
const NumRCFFiles = 12
const MaxRCFFileSize = 12*1024*1024


immutable BigFileIO
    num_bf::Int
    fds::Vector{Cint}
    pffd::Cint
end


function init_bigfile_io(datadir::String)
    all_b = NumRCFs * EachRCF
    quot, rem = divrem(all_b, EachBigFile)
    num_bf = quot + (rem > 0 ? 1 : 0)
    fds = Vector{Cint}(num_bf)
    for i = 1:num_bf
        fn = joinpath(datadir, "celeste_sdss-$i")
        fds[i] = ccall(:open, Cint, (Ptr{UInt8}, Cint, Cint), fn,
                       Base.Filesystem.JL_O_RDONLY, 0)
        if fds[i] == -1
            @show fn
            systemerror("open", true)
        end
    end

    fn = joinpath(datadir, "celeste_sdss-photofields")
    pffd = ccall(:open, Cint, (Ptr{UInt8}, Cint, Cint), fn,
                       Base.Filesystem.JL_O_RDONLY, 0)
    systemerror("open", pffd == -1)

    return BigFileIO(num_bf, fds, pffd)
end


function shutdown_bigfile_io(bfo::BigFileIO)
    ccall(:close, Cint, (Cint,), bfo.pffd)
    bfo.pffd = convert(Cint, -1)
    for i = 1:bfo.num_bf
        ccall(:close, Cint, (Cint,), bfo.fds[i])
    end
    bfo.num_bf = 0
    empty!(bfo.fds)
end

const fileOrder = Dict{Symbol, Int}(
    :fpMg       =>  1,
    :fpMi       =>  2,
    :fpMr       =>  3,
    :fpMu       =>  4,
    :fpMz       =>  5,
    :frameg     =>  6,
    :framei     =>  7,
    :framer     =>  8,
    :frameu     =>  9,
    :framez     => 10,
    :photoObj   => 11,
    :psField    => 12
)

struct RCFBundle
    images::Dict{Symbol, Vector{UInt8}}
    photofield_data::Vector{UInt8}
end

# No mutable state, so should be ok to manipulate GC state
@noinline function pread64(fd::Cint, data::Ptr{UInt8}, size::Csize_t, offset::Csize_t)
    nread = 0
    reqsize = size
    while nread < reqsize
        gc_state = ccall(:jl_gc_safe_enter, Int8, ())
        rdb = ccall(:pread64, Cint, (Cint, Ptr{UInt8}, Csize_t, Csize_t), fd, data, size, offset)
        ccall(:jl_gc_safe_leave, Void, (Int8,), gc_state)
        rdb <= 0 && break
        nread += rdb
        if rdb < size
            size -= rdb
            offset += rdb
        end
    end
    nread
end


function load_rcf_bundle(bfo::BigFileIO, rcf::RunCamcolField, rcf_idx_map, run_idx_map;
                         data_buf = zeros(UInt8, EachRCF + EachPhotoField))
    ridx = rcf_idx_map[rcf] - 1
    bfidx, bfofs = divrem(ridx * EachRCF, EachBigFile)
    bfidx += 1
    # We have a header, but it doesn't actually tell us how large the whole thing is, just slurp it all at once
    rsize = pread64(bfo.fds[bfidx], pointer(data_buf), Csize_t(EachRCF), Csize_t(bfofs))
    systemerror("pread", rsize != EachRCF)
    fofs = reinterpret(Int64, data_buf[1:(NumRCFFiles * sizeof(Int64))])

    # Now read the photofield data
    pidx = ((run_idx_map[rcf.run]-1) * NumCamcols * EachPhotoField) +
           ((rcf.camcol - 1) * EachPhotoField)
    rsize = pread64(bfo.pffd, pointer(data_buf) + EachRCF, Csize_t(EachPhotoField), Csize_t(pidx))
    systemerror("pread", rsize != EachPhotoField)
    photofield_data = data_buf[EachRCF+1:end]

    data = Dict{Symbol, Vector{UInt8}}()
    for (k, v) in fileOrder
        startidx = fofs[v]
        endidx = v == 12 ? EachRCF : fofs[v+1]
        (v == 1) && (@assert startidx == NumRCFFiles * sizeof(UInt64))
        data[k] = data_buf[startidx+1:endidx]
    end
    RCFBundle(data, photofield_data)
end
function load_rcf_bundles(bfo::BigFileIO, rcfs::Vector{RunCamcolField}, rcf_idx_map, run_idx_map;
                         data_buf = zeros(UInt8, EachRCF + EachPhotoField))
    [load_rcf_bundle(bfo, rcf, rcf_idx_map, run_idx_map) for rcf in rcfs]
end

function read_bigfile_photoobjs(rcf_bundles::Vector{RCFBundle};
                                duplicate_policy=:primary,
                                drop_quickly=false)
    @assert duplicate_policy == :primary || duplicate_policy == :first
    @assert duplicate_policy == :primary || length(fts) == 1

    if length(rcf_bundles) == 0
        return CatalogEntry[]
    end

    rawcatalogs = Vector{Dict}(length(rcf_bundles))
    for i in eachindex(rcf_bundles)
        rcf_bundle = rcf_bundles[i]
        po = read_photoobj(""; data=rcf_bundle.images[:photoObj])
        if !drop_quickly
            rawcatalogs[i] = po
        end
    end

    if drop_quickly
        return CatalogEntry[]
    end

    return assemble_catalog(rawcatalogs; duplicate_policy=duplicate_policy)
end

function load_raw_images_bigfile(rcf::RunCamcolField,
                                 rcf_bundle::RCFBundle;
                                 drop_quickly = false)
    gains = read_field_gains("", rcf.field; data = rcf_bundle.photofield_data)
    psffile = FITSIO.FITS(rcf_bundle.images[:psField])
    raw_images = Vector{RawImage}(5)
    for (b, band) in enumerate(['u', 'g', 'r', 'i', 'z'])
        # load image data
        r = parse_image(rcf, b, "", "", psffile, band, gains;
            frame_data = rcf_bundle.images[Symbol("frame$band")],
            mask_data = rcf_bundle.images[Symbol("fpM$band")])
        if !drop_quickly
            raw_images[b] = r
        end
    end
    raw_images
end

function load_raw_images_bigfile(rcfs::Vector{RunCamcolField},
                                 rcf_bundles::Vector{RCFBundle};
                                 drop_quickly = false)
    mapreduce(data->load_raw_images_bigfile(data...; drop_quickly=drop_quickly), vcat, zip(rcfs, rcf_bundles))
end


# IO Strategies
abstract type IOStrategy end

struct PlainFITSStrategy <: IOStrategy
    stagedir::String
    # Could be a string or rcf -> stagedir function
    rcf_stagedir
    slurp::Bool
end
PlainFITSStrategy(stagedir::String, slurp::Bool = false) = PlainFITSStrategy(stagedir, stagedir, slurp)

struct BigFileStrategy <: IOStrategy
    stagedir::String
    bfo::BigFileIO
    rcf_idx_map
    run_idx_map
end

# We load data on demand, do nothing here
preload_rcfs(::PlainFITSStrategy, rcfs) = [nothing for rcf in rcfs]
preload_rcfs(strategy::BigFileStrategy, rcfs) = load_rcf_bundles(strategy.bfo, rcfs, strategy.rcf_idx_map, strategy.run_idx_map)

read_photoobj(strategy::PlainFITSStrategy, rcf, state; duplicate_policy=:primary, drop_quickly::Bool=false) =
    read_photoobj_files([rcf],strategy.rcf_stagedir,
        duplicate_policy = duplicate_policy,
        slurp = strategy.slurp,
        drop_quickly = drop_quickly)

read_photoobj(strategy::BigFileStrategy, rcf, rcf_bundle; duplicate_policy=:primary, drop_quickly::Bool=false) =
    read_bigfile_photoobjs([rcf_bundle],
        duplicate_policy = duplicate_policy,
        drop_quickly = drop_quickly)

load_raw_images(strategy::PlainFITSStrategy, rcfs, states, drop_quickly::Bool = false) =
    SDSSIO.load_raw_images(rcfs, strategy.rcf_stagedir, strategy.slurp, drop_quickly)

load_raw_images(strategy::BigFileStrategy, rcfs, states, drop_quickly::Bool = false) =
    SDSSIO.load_raw_images_bigfile(rcfs, states; drop_quickly=drop_quickly)

"""
Read a SDSS run/camcol/field into an array of Images.
"""
function load_field_images(strategy::IOStrategy, rcfs, states, drop_quickly::Bool = false)
    raw_images = SDSSIO.load_raw_images(strategy, rcfs, states, drop_quickly)

    N = length(raw_images)
    images = Vector{Image}(N)

    drop_quickly && return images

    #Threads.@threads for n in 1:N
    for n in 1:N
        try
            images[n] = convert(Image, raw_images[n])
        catch exc
            Log.exception(exc)
            rethrow()
        end
    end

    return images
end


end  # module
