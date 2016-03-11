module SkyImages

using ..Types
import SloanDigitalSkySurvey: PSF, SDSS, WCSUtils
import SloanDigitalSkySurvey.PSF.get_psf_at_point

import WCS
import DataFrames
import ..ElboDeriv # For stitch_object_tiles
import FITSIO
import ..Util

export load_stamp_blob, load_sdss_blob, crop_image!,
       get_psf_at_point,
       convert_catalog_to_celeste, load_stamp_catalog,
       break_blob_into_tiles, break_image_into_tiles,
       get_local_sources,
       stitch_object_tiles

# The default mask planes are those used in the astrometry.net code.
const DEFAULT_MASK_PLANES = ["S_MASK_INTERP",  # bad pixel (was interpolated)
                             "S_MASK_SATUR",  # saturated
                             "S_MASK_CR",  # cosmic ray
                             "S_MASK_GHOST"]  # electronics artifacts

"""
interp_sky(data, xcoords, ycoords)

Interpolate the 2-d array `data` at the grid of array coordinates spanned
by the vectors `xcoords` and `ycoords` using bilinear interpolation.
The output array will have size `(length(xcoords), length(ycoords))`.
For example, if `x[1] = 3.3` and `y[2] = 4.7`, the element `out[1, 2]`
will be a result of linear interpolation between the values
`data[3:4, 4:5]`.

Coordinates should not extend more than 1 element past size of data.
"""
function interp_sky{T, S}(data::Array{T, 2}, xcoords::Vector{S},
                          ycoords::Vector{S})
    # We assume below that 0 <= floor(x) <= size(data, 1)
    # and similarly for y. Check this.
    for xc in xcoords
        if xc < zero(S) || xc >= size(data, 1) + 1
            error("x coordinates out of bounds")
        end
    end
    for yc in ycoords
        if yc < zero(S) || yc >= size(data, 2) + 1
            error("y coordinates out of bounds")
        end
    end

    out = Array(T, length(xcoords), length(ycoords))
    for j=1:length(ycoords)
        y0 = floor(Int, ycoords[j])
        y1 = y0 + 1
        yw0 = ycoords[j] - y0
        yw1 = one(S) - yw0
        y0 = max(y0, 1)
        y1 = min(y1, size(data, 2))
        for i=1:length(xcoords)
            x0 = floor(Int, xcoords[i])
            x1 = x0 + 1
            xw0 = xcoords[i] - x0
            xw1 = one(S) - xw0
            x0 = max(x0, 1)
            x1 = min(x1, size(data, 1))
            @inbounds out[i, j] = (xw0 * yw0 * data[x0, y0] +
                                   xw1 * yw0 * data[x1, y0] +
                                   xw0 * yw1 * data[x0, y1] +
                                   xw1 * yw1 * data[x1, y1])
        end
    end
    return out
end

function load_raw_field(field_dir, run_num, camcol_num, field_num, b, gain)
    b_letter = band_letters[b]

    fname = "$field_dir/frame-$b_letter-$run_num-$camcol_num-$field_num.fits"
    @assert(isfile(fname), "Cannot find $(fname)")
    f = FITSIO.FITS(fname)

    # This is the sky-subtracted and calibrated image.
    image = read(f[1])::Array{Float32, 2}

    # Read in the sky background.
    sky_small = squeeze(read(f[3], "ALLSKY"), 3)::Array{Float32, 2}
    sky_x = vec(read(f[3], "XINTERP"))::Vector{Float32}
    sky_y = vec(read(f[3], "YINTERP"))::Vector{Float32}

    # convert sky interpolation coordinates from 0-indexed to 1-indexed
    for i=1:length(sky_x)
        sky_x[i] += 1.0f0
    end
    for i=1:length(sky_y)
        sky_y[i] += 1.0f0
    end

    # Load the WCSTransform
    header_str = FITSIO.read_header(f[1], ASCIIString)::ASCIIString
    wcs = WCS.from_header(header_str)[1]

    # Calibration vector
    calibration = read(f[2])::Vector{Float32}

    close(f)

    # interpolate to full sky image
    sky = interp_sky(sky_small, sky_x, sky_y)

    # Convert image to raw electron counts.  Note that these may not
    # be close to integers due to the analog to digital conversion
    # process in the telescope.
    for j=1:size(image, 2), i=1:size(image, 1)
        image[i, j] = gain * (image[i, j] / calibration[i] + sky[i, j])
    end

    image, calibration, sky, wcs
end

"""
load_sdss_field_gains(fname, fieldnum)

Return the image gains for field number `fieldnum` in an SDSS
\"photoField\" file `fname`.
"""
function load_sdss_field_gains(fname, fieldnum::Integer)

    f = FITSIO.FITS(fname)
    fieldnums = read(f[2], "FIELD")::Vector{Int32}
    gains = read(f[2], "GAIN")::Array{Float32, 2}
    close(f)

    # Find first occurance of `fieldnum` and return the corresponding gain.
    for i=1:length(fieldnums)
        fieldnums[i] == fieldnum && return gains[:, i]
    end

    error("field number $fieldnum not found in file: $fname")
end


"""
load_sdss_mask(fname[, mask_planes])

Read a \"fpM\"-format SDSS file and return masked image ranges,
based on `mask_planes`. Returns two `Vector{UnitRange{Int}}`,
giving the range of x and y indicies to be masked.
"""
function load_sdss_mask(fname, mask_planes=DEFAULT_MASK_PLANES)
    f = FITSIO.FITS(fname)

    # The last (12th) HDU contains a key describing what each of the
    # other HDUs are. Use this to find the indicies of all the relevant
    # HDUs (those with attributeName matching a value in `mask_planes`).
    value = read(f[12], "Value")::Vector{Int32}
    def = read(f[12], "defName")::Vector{ASCIIString}
    attribute = read(f[12], "attributeName")::Vector{ASCIIString}

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
load_sdss_psf(fname, b)

Read a `RawPSFComponents` for band number `b` from the SDSS \"psField\"
file `fname`. `b` must be in range 1:5.
"""
function load_sdss_psf(fname, b::Integer)
    @assert b in 1:5

    f = FITSIO.FITS(fname)
    hdu = f[b + 1]
    nrows = FITSIO.read_key(hdu, "NAXIS2")[1]::Int
    nrow_b = (read(hdu, "nrow_b")::Vector{Int32})[1]
    ncol_b = (read(hdu, "ncol_b")::Vector{Int32})[1]
    rnrow = (read(hdu, "rnrow")::Vector{Int32})[1]
    rncol = (read(hdu, "rncol")::Vector{Int32})[1]
    cmat_raw = read(hdu, "c")::Array{Float32, 3}
    rrows_raw = read(hdu, "rrows")::Array{Array{Float32,1},1}
    close(f)

    # Only the first (nrow_b, ncol_b) submatrix of cmat is used for reasons obscure
    # to the author.
    cmat = Array(Float64, nrow_b, ncol_b, size(cmat_raw, 3))
    for k=1:size(cmat_raw, 3), j=1:nrow_b, i=1:ncol_b
        cmat[i, j, k] = cmat_raw[i, j, k]
    end

    # convert rrows to Array{Float64, 2}, assuming each row is the same length.
    rrows = Array(Float64, length(rrows_raw[1]), length(rrows_raw))
    for i=1:length(rrows_raw)
        rrows[:, i] = rrows_raw[i]
    end

    return PSF.RawPSFComponents(rrows, rnrow, rncol, cmat)
end


"""
Load a stamp catalog.
"""
function load_stamp_catalog(cat_dir, stamp_id, blob; match_blob=false)
    df = SDSS.load_stamp_catalog_df(cat_dir, stamp_id, blob,
                                    match_blob=match_blob)
    df[:objid] = [ string(s) for s=1:size(df)[1] ]
    convert_catalog_to_celeste(df, blob, match_blob=match_blob)
end


"""
Convert a dataframe catalog (e.g. as returned by
SloanDigitalSkySurvey.SDSS.load_catalog_df) to an array of Celeste CatalogEntry
objects.

Args:
  - df: The dataframe output of SloanDigitalSkySurvey.SDSS.load_catalog_df
  - blob: The blob that the catalog corresponds to
  - match_blob: If false, changes the direction of phi to match tractor,
                not Celeste.
"""
function convert_catalog_to_celeste(
  df::DataFrames.DataFrame, blob::Array{Image, 1}; match_blob=false)
    function row_to_ce(row)
        x_y = [row[1, :ra], row[1, :dec]]
        star_fluxes = zeros(5)
        gal_fluxes = zeros(5)
        fracs_dev = [row[1, :frac_dev], 1 - row[1, :frac_dev]]
        for b in 1:length(band_letters)
            bl = band_letters[b]
            psf_col = symbol("psfflux_$bl")

            # TODO: How can there be negative fluxes?
            star_fluxes[b] = max(row[1, psf_col], 1e-6)

            dev_col = symbol("devflux_$bl")
            exp_col = symbol("expflux_$bl")
            gal_fluxes[b] += fracs_dev[1] * max(row[1, dev_col], 1e-6) +
                             fracs_dev[2] * max(row[1, exp_col], 1e-6)
        end

        fits_ab = fracs_dev[1] > .5 ? row[1, :ab_dev] : row[1, :ab_exp]
        fits_phi = fracs_dev[1] > .5 ? row[1, :phi_dev] : row[1, :phi_exp]
        fits_theta = fracs_dev[1] > .5 ? row[1, :theta_dev] : row[1, :theta_exp]

        # tractor defines phi as -1 * the phi catalog for some reason.
        if !match_blob
            fits_phi *= -1.
        end

        re_arcsec = max(fits_theta, 1. / 30)  # re = effective radius
        re_pixel = re_arcsec / 0.396

        phi90 = 90 - fits_phi
        phi90 -= floor(phi90 / 180) * 180
        phi90 *= (pi / 180)

        CatalogEntry(x_y, row[1, :is_star], star_fluxes,
            gal_fluxes, row[1, :frac_dev], fits_ab, phi90, re_pixel,
            row[1, :objid])
    end

    CatalogEntry[row_to_ce(df[i, :]) for i in 1:size(df, 1)]
end


"""
Load a stamp into a Celeste blob.
"""
function load_stamp_blob(stamp_dir, stamp_id)
    function fetch_image(b)
        band_letter = band_letters[b]
        filename = "$stamp_dir/stamp-$band_letter-$stamp_id.fits"

        fits = FITSIO.FITS(filename)
        hdr = FITSIO.read_header(fits[1])
        original_pixels = read(fits[1])
        dn = original_pixels / hdr["CALIB"] + hdr["SKY"]
        nelec_f32 = round(dn * hdr["GAIN"])
        nelec = convert(Array{Float64}, nelec_f32)

        header_str = FITSIO.read_header(fits[1], ASCIIString)
        wcs = WCS.from_header(header_str)[1]
        close(fits)

        alphaBar = [hdr["PSF_P0"]; hdr["PSF_P1"]; hdr["PSF_P2"]]
        xiBar = [
            [hdr["PSF_P3"]  hdr["PSF_P4"]];
            [hdr["PSF_P5"]  hdr["PSF_P6"]];
            [hdr["PSF_P7"]  hdr["PSF_P8"]]]'

        tauBar = Array(Float64, 2, 2, 3)
        tauBar[:,:,1] = [[hdr["PSF_P9"] hdr["PSF_P11"]];
                         [hdr["PSF_P11"] hdr["PSF_P10"]]]
        tauBar[:,:,2] = [[hdr["PSF_P12"] hdr["PSF_P14"]];
                         [hdr["PSF_P14"] hdr["PSF_P13"]]]
        tauBar[:,:,3] = [[hdr["PSF_P15"] hdr["PSF_P17"]];
                         [hdr["PSF_P17"] hdr["PSF_P16"]]]

        psf = [PsfComponent(alphaBar[k], xiBar[:, k],
                            tauBar[:, :, k]) for k in 1:3]

        H, W = size(original_pixels)
        iota = hdr["GAIN"] / hdr["CALIB"]
        epsilon = hdr["SKY"] * hdr["CALIB"]

        run_num = round(Int, hdr["RUN"])
        camcol_num = round(Int, hdr["CAMCOL"])
        field_num = round(Int, hdr["FIELD"])

        Image(H, W, nelec, b, wcs, epsilon, iota, psf,
              run_num, camcol_num, field_num)
    end

    blob = map(fetch_image, 1:5)
end


"""
Read a blob from SDSS.

Args:
 - field_dir: The directory of the file
 - run_num: The run number
 - camcol_num: The camcol number
 - field_num: The field number

Returns:
 - A blob (array of Image objects).
"""
function load_sdss_blob(field_dir, run_num, camcol_num, field_num;
  mask_planes =
    Set(["S_MASK_INTERP", "S_MASK_SATUR", "S_MASK_CR", "S_MASK_GHOST"]))

    # Read gain for each band from "photoField" file.
    pf_fname = "$field_dir/photoField-$run_num-$camcol_num.fits"
    @assert(isfile(pf_fname), "Cannot find $(pf_fname)")
    gains = load_sdss_field_gains(pf_fname, parse(Int, field_num))

    blob = Array(Image, 5)
    for b=1:5
        print("Reading band $b image data... ")
        nelec, calib_col, sky_image, wcs =
            load_raw_field(field_dir, run_num, camcol_num,
                           field_num, b, gains[b])

        letter = band_letters[b]
        mask_fname = joinpath(field_dir,
                              "fpM-$run_num-$letter$camcol_num-$field_num.fit")
        @assert(isfile(mask_fname), "Cannot find mask file $(mask_fname)")

        print("masking image... ")
        mask_xranges, mask_yranges = load_sdss_mask(mask_fname, mask_planes)

        # apply mask
        for i=1:length(mask_xranges)
            nelec[mask_xranges[i], mask_yranges[i]] = NaN
        end

        H = size(nelec, 1)
        W = size(nelec, 2)

        # For now, use the median noise and sky image.  Here,
        # epsilon * iota needs to be in units comparable to nelec
        # electron counts.
        # Note that each are actuall pretty variable.
        iota = convert(Float64, gains[b] / median(calib_col))
        epsilon = convert(Float64, median(sky_image) * median(calib_col))
        epsilon_mat = Array(Float64, H, W)
        iota_vec = Array(Float64, H)
        for h=1:H
            iota_vec[h] = gains[b] / calib_col[h]
            for w=1:W
                epsilon_mat[h, w] = sky_image[h, w] * calib_col[h]
            end
        end

        # Load and fit the psf.
        print("reading psf... ")
        psf_fname = "$field_dir/psField-$run_num-$camcol_num-$field_num.fit"
        @assert(isfile(psf_fname), "Cannot find mask file $(psf_fname)")
        raw_psf_comp = load_sdss_psf(psf_fname, b)

        # For now, evaluate the psf at the middle of the image.
        psf_point_x = H / 2
        psf_point_y = W / 2

        raw_psf = get_psf_at_point(psf_point_x, psf_point_y, raw_psf_comp);
        psf = fit_raw_psf_for_celeste(raw_psf);

        println("done.")

        # Set it to use a constant background but include the non-constant data.
        blob[b] = Image(H, W, nelec, b, wcs, epsilon, iota, psf,
                      parse(Int, run_num),
                      parse(Int, camcol_num),
                      parse(Int, field_num),
                      false, epsilon_mat, iota_vec, raw_psf_comp)
    end

    blob
end


"""
Crop an image in place to a (2 * width) x (2 * width) - pixel square centered
at the world coordinates wcs_center.
Args:
  - blob: The field to crop
  - width: The width in pixels of each quadrant
  - wcs_center: A location in world coordinates (e.g. the location of a
                celestial body)

Returns:
  - A tiled blob with a single tile in each image centered at wcs_center.
    This can be used to investigate a certain celestial object in a single
    tiled blob, for example.
"""
function crop_blob_to_location(
  blob::Array{Image, 1},
  width::Union{Float64, Int},
  wcs_center::Vector{Float64})
    @assert length(wcs_center) == 2
    @assert width > 0

    tiled_blob = Array(TiledImage, length(blob))
    for b=1:5
        # Get the pixels that are near enough to the wcs_center.
        pix_center = WCSUtils.world_to_pix(blob[b].wcs, wcs_center)
        h_min = max(floor(Int, pix_center[1] - width), 1)
        h_max = min(ceil(Int, pix_center[1] + width), blob[b].H)
        sub_rows_h = h_min:h_max

        w_min = max(floor(Int, (pix_center[2] - width)), 1)
        w_max = min(ceil(Int, pix_center[2] + width), blob[b].W)
        sub_rows_w = w_min:w_max
        tiled_blob[b] = fill(ImageTile(blob[b], sub_rows_h, sub_rows_w), 1, 1)
    end
    tiled_blob
end


############################################
# PSF functions


function fit_raw_psf_for_celeste(raw_psf::Matrix{Float64})
  # TODO: this is very slow, and we should do it in Celeste rather
  # than rely on Optim.
  opt_result, mu_vec, sigma_vec, weight_vec =
    PSF.fit_psf_gaussians_least_squares(raw_psf, K=psf_K);
  PsfComponent[ PsfComponent(weight_vec[k], mu_vec[k], sigma_vec[k])
    for k=1:psf_K]
end

"""
Return an image of a Celeste GMM PSF evaluated at rows, cols.

Args:
 - psf_array: The PSF to be evaluated as an array of PsfComponent
 - rows: The rows in the image (usually in pixel coordinates)
 - cols: The column in the image (usually in pixel coordinates)

 Returns:
  - The PSF values at rows and cols.  The default size is the same as
    that returned by get_psf_at_point applied to FITS header values.

Note that the point in the image at which the PSF is evaluated --
that is, the center of the image returned by this function -- is
already implicit in the value of psf_array.
"""
function get_psf_at_point(psf_array::Array{PsfComponent, 1};
                          rows=collect(-25:25), cols=collect(-25:25))

    function get_psf_value(psf::PsfComponent, row::Float64, col::Float64)
        x = Float64[row, col] - psf.xiBar
        exp_term = exp(-0.5 * x' * psf.tauBarInv * x - 0.5 * psf.tauBarLd)
        (psf.alphaBar * exp_term / (2 * pi))[1]
    end

    Float64[
      sum([ get_psf_value(psf, float(row), float(col)) for psf in psf_array ])
        for row in rows, col in cols ]
end


"""
Get the PSF located at a particular world location in an image.

Args:
  - world_loc: A location in world coordinates.
  - img: An Image

Returns:
  - An array of PsfComponent objects that represents the PSF as a mixture
    of Gaussians.
"""
function get_source_psf(world_loc::Vector{Float64}, img::Image)
    # Some stamps or simulated data have no raw psf information.  In that case,
    # just use the psf from the image.
    if size(img.raw_psf_comp.rrows) == (0, 0)
      return img.psf
    else
      pixel_loc = WCSUtils.world_to_pix(img.wcs, world_loc)
      raw_psf =
        PSF.get_psf_at_point(pixel_loc[1], pixel_loc[2], img.raw_psf_comp);
      return fit_raw_psf_for_celeste(raw_psf);
    end
end


#######################################
# Tiling functions

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
  ImageTile[ ImageTile(hh, ww, img, tile_width) for hh=1:HH, ww=1:WW ]
end


"""
Break a blob into tiles.
"""
function break_blob_into_tiles(blob::Blob, tile_width::Int)
  [ break_image_into_tiles(img, tile_width) for img in blob ]
end


#######################################
# Functions for matching sources to tiles.

# A pixel circle maps locally to a world ellipse.  Return the major
# axis of that ellipse.
function pixel_radius_to_world(pix_radius::Float64,
                               wcs_jacobian::Matrix{Float64})
  pix_radius / minimum(abs(eig(wcs_jacobian)[1]));
end


# A world circle maps locally to a pixel ellipse.  Return the major
# axis of that ellipse.
function world_radius_to_pixel(world_radius::Float64,
                               wcs_jacobian::Matrix{Float64})
  world_radius * maximum(abs(eig(wcs_jacobian)[1]));
end


import SloanDigitalSkySurvey.WCSUtils.world_to_pix
world_to_pix{T <: Number}(patch::SkyPatch, world_loc::Vector{T}) =
    world_to_pix(patch.wcs_jacobian, patch.center, patch.pixel_center,
                 world_loc)


"""
A fast function to determine which sources might belong to which tiles.

Args:
  - tiles: A TiledImage
  - patches: A vector of patches (e.g. for a particular band)

Returns:
  - An array (over tiles) of a vector of candidate
    source patches.  If a patch is a candidate, it may be within the patch radius
    of a point in the tile, though it might not.
"""
function local_source_candidates(
    tiles::TiledImage, patches::Vector{SkyPatch})

  # Get the largest size of the pixel ellipse defined by the patch
  # world coordinate circle.
  patch_pixel_radii =
    Float64[patches[s].radius * maximum(abs(eig(patches[s].wcs_jacobian)[1]))
            for s=1:length(patches)];

  candidates = fill(Int[], size(tiles));
  for h=1:size(tiles)[1], w=1:size(tiles)[2]
    # Find the patches that are less than the radius plus diagonal from the
    # center of the tile.  These are candidates for having some
    # overlap with the tile.
    tile = tiles[h, w]
    tile_center = [ mean(tile.h_range), mean(tile.w_range)]
    tile_diag = (0.5 ^ 2) * (tile.h_width ^ 2 + tile.w_width ^ 2)
    patch_distances =
      [ sum((tile_center .- patches[s].pixel_center) .^ 2) for
        s=1:length(patches)]
    candidates[h, w] =
      find(patch_distances .<= (tile_diag .+ patch_pixel_radii) .^ 2)
  end

  candidates
end


"""
Args:
  - tile: An ImageTile (containing tile coordinates)
  - patches: A vector of SkyPatch objects to be matched with the tile.

Returns:
  - A vector of source ids (from 1 to length(patches)) that influence
    pixels in the tile.  A patch influences a tile if
    there is any overlap in their squares of influence.
"""
function get_local_sources(tile::ImageTile, patches::Vector{SkyPatch})

    tile_sources = Int[]
    tile_center = Float64[mean(tile.h_range), mean(tile.w_range)]

    for patch_index in 1:length(patches)
      patch = patches[patch_index]
      patch_radius_px = world_radius_to_pixel(patch.radius, patch.wcs_jacobian)

      # This is a "ball" in the infinity norm.
      if (abs(tile_center[1] - patch.pixel_center[1]) <
          patch_radius_px + 0.5 * tile.h_width) &&
         (abs(tile_center[2] - patch.pixel_center[2]) <
          patch_radius_px + 0.5 * tile.w_width)
        push!(tile_sources, patch_index)
      end
    end

    tile_sources
end


"""
Return a vector of (h, w) indices of tiles that contain this source.
"""
function find_source_tiles(s::Int, b::Int, mp::ModelParams)
  [ ind2sub(size(mp.tile_sources[b]), ind) for ind in
    find([ s in sources for sources in mp.tile_sources[b]]) ]
end

"""
Combine the tiles associated with a single source into an image.

Args:
  s: The source index
  b: The band
  mp: The ModelParams object
  tiled_blob: The original tiled blob
  predicted: If true, render the object based on the values in ModelParams.
             Otherwise, show the image from tiled_blob.

Returns:
  A matrix of pixel values for the particular object using only tiles in
  which it is found according to the ModelParams tile_sources field.  Pixels
  from tiles that do not have this source will be marked as 0.0.
"""
function stitch_object_tiles(
    s::Int, b::Int, mp::ModelParams{Float64}, tiled_blob::TiledBlob;
    predicted::Bool=false)

  H, W = size(tiled_blob[b])
  has_s = Bool[ s in mp.tile_sources[b][h, w] for h=1:H, w=1:W ];
  tiles_s = tiled_blob[b][has_s];
  tile_sources_s = mp.tile_sources[b][has_s];
  h_range = Int[typemax(Int), typemin(Int)]
  w_range = Int[typemax(Int), typemin(Int)]
  print("Stitching...")
  for tile in tiles_s
    print(".")
    h_range[1] = min(minimum(tile.h_range), h_range[1])
    h_range[2] = max(maximum(tile.h_range), h_range[2])
    w_range[1] = min(minimum(tile.w_range), w_range[1])
    w_range[2] = max(maximum(tile.w_range), w_range[2])
  end

  image_s = fill(0.0, diff(h_range)[1] + 1, diff(w_range)[1] + 1);
  for tile_ind in 1:length(tiles_s)
    tile = tiles_s[tile_ind]
    tile_sources = tile_sources_s[tile_ind]
    image_s[tile.h_range - h_range[1] + 1, tile.w_range - w_range[1] + 1] =
      predicted ?
      ElboDeriv.tile_predicted_image(tile, mp, Int[ s ], include_epsilon=false):
      tile.pixels
  end
  println("Done.")
  image_s, h_range, w_range
end


end  # module
