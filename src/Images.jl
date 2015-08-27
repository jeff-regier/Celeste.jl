module Images

VERSION < v"0.4.0-dev" && using Docile
using CelesteTypes
import SloanDigitalSkySurvey: SDSS
import SloanDigitalSkySurvey: WCS
import SloanDigitalSkySurvey: PSF
import SloanDigitalSkySurvey.PSF.get_psf_at_point

import WCSLIB
import DataFrames
import FITSIO
import GaussianMixtures
import Grid
import PSF
import Util
import WCS

export load_stamp_blob, load_sdss_blob, crop_image!, test_catalog_entry_in_image
export convert_gmm_to_celeste, get_psf_at_point
export convert_catalog_to_celeste, load_stamp_catalog


function load_stamp_catalog(cat_dir, stamp_id, blob; match_blob=false)
    df = SDSS.load_stamp_catalog_df(cat_dir, stamp_id, blob, match_blob=match_blob)
    convert_catalog_to_celeste(df, blob, match_blob=match_blob)
end


@doc """
Convert a dataframe catalog (e.g. as returned by
SloanDigitalSkySurvey.SDSS.load_catalog_df) to an array of Celeste CatalogEntry
objects.

Args:
  - df: The dataframe output of SloanDigitalSkySurvey.SDSS.load_catalog_df
  - blob: The blob that the catalog corresponds to
  - match_blob: If false, changes the direction of phi to match tractor,
                not Celeste.
""" ->
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
            star_fluxes[b] = row[1, psf_col]

            dev_col = symbol("devflux_$bl")
            exp_col = symbol("expflux_$bl")
            gal_fluxes[b] += fracs_dev[1] * row[1, dev_col] +
                             fracs_dev[2] * row[1, exp_col]
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
            gal_fluxes, row[1, :frac_dev], fits_ab, phi90, re_pixel)
    end

    CatalogEntry[row_to_ce(df[i, :]) for i in 1:size(df, 1)]
end


@doc """
Load a stamp into a Celeste blob.
""" ->
function load_stamp_blob(stamp_dir, stamp_id)
    function fetch_image(b)
        band_letter = band_letters[b]
        filename = "$stamp_dir/stamp-$band_letter-$stamp_id.fits"

        fits = FITSIO.FITS(filename)
        hdr = FITSIO.read_header(fits[1])
        original_pixels = read(fits[1])
        dn = original_pixels / hdr["CALIB"] + hdr["SKY"]
        nelec = float(int(dn * hdr["GAIN"]))

        header_str = FITSIO.read_header(fits[1], ASCIIString)
        ((wcs,),nrejected) = WCSLIB.wcspih(header_str)
        close(fits)

        alphaBar = [hdr["PSF_P0"], hdr["PSF_P1"], hdr["PSF_P2"]]
        xiBar = [
            [hdr["PSF_P3"]  hdr["PSF_P4"]],
            [hdr["PSF_P5"]  hdr["PSF_P6"]],
            [hdr["PSF_P7"]  hdr["PSF_P8"]]]'

        tauBar = Array(Float64, 2, 2, 3)
        tauBar[:,:,1] = [[hdr["PSF_P9"] hdr["PSF_P11"]],
                         [hdr["PSF_P11"] hdr["PSF_P10"]]]
        tauBar[:,:,2] = [[hdr["PSF_P12"] hdr["PSF_P14"]],
                         [hdr["PSF_P14"] hdr["PSF_P13"]]]
        tauBar[:,:,3] = [[hdr["PSF_P15"] hdr["PSF_P17"]],
                         [hdr["PSF_P17"] hdr["PSF_P16"]]]

        psf = [PsfComponent(alphaBar[k], xiBar[:, k],
                            tauBar[:, :, k]) for k in 1:3]

        H, W = size(original_pixels)
        iota = hdr["GAIN"] / hdr["CALIB"]
        epsilon = hdr["SKY"] * hdr["CALIB"]

        run_num = int(hdr["RUN"])
        camcol_num = int(hdr["CAMCOL"])
        field_num = int(hdr["FIELD"])

        Image(H, W, nelec, b, wcs, epsilon, iota, psf, run_num, camcol_num, field_num)
    end

    blob = map(fetch_image, 1:5)
end


@doc """
Read a blob from SDSS.

Args:
 - field_dir: The directory of the file
 - run_num: The run number
 - camcol_num: The camcol number
 - field_num: The field number

Returns:
 - A blob (array of Image objects).
""" ->
function load_sdss_blob(field_dir, run_num, camcol_num, field_num)

    band_gain, band_dark_variance =
      SDSS.load_photo_field(field_dir, run_num, camcol_num, field_num)

    blob = Array(Image, 5)
    for b=1:5
        print("Reading band $b image data...")
        nelec, calib_col, sky_grid, sky_x, sky_y, sky_image, wcs =
            SDSS.load_raw_field(field_dir, run_num, camcol_num, field_num, b, band_gain[b]);

        print("Masking image...")
        SDSS.mask_image!(nelec, field_dir, run_num, camcol_num, field_num, b);
        println("done.")
        H = size(nelec, 1)
        W = size(nelec, 2)

        # For now, use the median noise and sky image.  Here,
        # epsilon * iota needs to be in units comparable to nelec electron counts.
        # Note that each are actuall pretty variable.
        iota = convert(Float64, band_gain[b] / median(calib_col))
        epsilon = convert(Float64, median(sky_image) * median(calib_col))
        epsilon_mat = Array(Float64, H, W)
        iota_vec = Array(Float64, H)
        for h=1:H
            iota_vec[h] = band_gain[b] / calib_col[h]
            for w=1:W
                epsilon_mat[h, w] = sky_image[h, w] * calib_col[h]
            end
        end

        # Load and fit the psf.
        println("reading psf...")
        raw_psf_comp = SDSS.load_psf_data(field_dir, run_num, camcol_num, field_num, b);

        # For now, evaluate the psf at the middle of the image.
        psf_point_x = H / 2
        psf_point_y = W / 2

        raw_psf = get_psf_at_point(psf_point_x, psf_point_y, raw_psf_comp);
        psf_gmm, scale = PSF.fit_psf_gaussians(raw_psf);
        psf = convert_gmm_to_celeste(psf_gmm, scale)

        # Set it to use a constant background but include the non-constant data.
        blob[b] = Image(H, W,
                        nelec, b, wcs,
                        epsilon, iota, psf,
                        int(run_num), int(camcol_num), int(field_num),
                        true, epsilon_mat, iota_vec, raw_psf_comp)
    end

    blob
end


@doc """
Crop an image in place to a (2 * width) x (2 * width) - pixel square centered
at the world coordinates wcs_center.
Args:
  - blob: The field to crop
  - width: The width in pixels of each quadrant
  - wcs_center: A location in world coordinates (e.g. the location of a celestial body)
""" ->
function crop_image!(
  blob::Array{Image, 1}, width::Float64, wcs_center::Vector{Float64})
    @assert length(wcs_center) == 2
    @assert width > 0

    # Get the original world coordinate centers.
    original_crpix_band =
      Float64[unsafe_load(blob[b].wcs.crpix, i) for i=1:2, b=1:5];

    x_ranges = zeros(2, 5)
    y_ranges = zeros(2, 5)
    for b=1:5
        # Get the pixels that are near enough to the wcs_center.
        obj_loc_pix = WCS.world_to_pixel(blob[b].wcs, wcs_center)
        sub_rows_x = floor(obj_loc_pix[1] - width):ceil(obj_loc_pix[1] + width)
        sub_rows_y = floor(obj_loc_pix[2] - width):ceil(obj_loc_pix[2] + width)
        x_min = minimum(collect(sub_rows_x))
        y_min = minimum(collect(sub_rows_y))
        x_max = maximum(collect(sub_rows_x))
        y_max = maximum(collect(sub_rows_y))
        x_ranges[:, b] = Float64[x_min, x_max]
        y_ranges[:, b] = Float64[y_min, y_max]

        # Crop the image down to the selected pixels.
        # Re-center the WCS coordinates
        crpix = original_crpix_band[:, b]
        unsafe_store!(blob[b].wcs.crpix, crpix[1] - x_min + 1, 1)
        unsafe_store!(blob[b].wcs.crpix, crpix[2] - y_min + 1, 2)

        blob[b].pixels = blob[b].pixels[sub_rows_x, sub_rows_y]
        blob[b].H = size(blob[b].pixels, 1)
        blob[b].W = size(blob[b].pixels, 2)
        blob[b].iota_vec = blob[b].iota_vec[x_min:x_max]
        blob[b].epsilon_mat = blob[b].epsilon_mat[x_min:x_max, y_min:y_max]
    end

    x_ranges, y_ranges
end

@doc """
Check whether the center of a celestial body is in any of the frames of an image.
Args:
  - blob: The image to check
  - wcs_loc: A location in world coordinates (e.g. the location of a celestial body)
Returns:
  - Whether the pixel wcs_loc lies within any of the image's fields.
""" ->
function test_catalog_entry_in_image(
  blob::Array{Image, 1}, wcs_loc::Array{Float64, 1})
    for b=1:5
        pixel_loc = WCS.world_to_pixel(blob[b].wcs, wcs_loc)
        if (1 <= pixel_loc[1] <= blob[b].H) && (1 <= pixel_loc[2] <= blob[b].W)
            return true
        end
    end
    return false
end


@doc """
Convert a GaussianMixtures.GMM object to an array of Celect PsfComponents.

Args:
 - gmm: A GaussianMixtures.GMM object (e.g. as returned by fit_psf_gaussians)

 Returns:
  - An array of PsfComponent objects.
""" ->
function convert_gmm_to_celeste(gmm::GaussianMixtures.GMM, scale::Float64)
    function convert_gmm_component_to_celeste(gmm::GaussianMixtures.GMM, d)
        CelesteTypes.PsfComponent(scale * gmm.w[d],
            collect(GaussianMixtures.means(gmm)[d, :]),
            GaussianMixtures.covars(gmm)[d])
    end

    CelesteTypes.PsfComponent[ convert_gmm_component_to_celeste(gmm, d) for d=1:gmm.n ]
end


@doc """
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
""" ->
function get_psf_at_point(psf_array::Array{CelesteTypes.PsfComponent, 1};
                          rows=collect(-25:25), cols=collect(-25:25))

    function get_psf_value(
      psf::CelesteTypes.PsfComponent, row::Float64, col::Float64)
        x = Float64[row, col] - psf.xiBar
        exp_term = exp(-0.5 * x' * psf.tauBarInv * x - 0.5 * psf.tauBarLd)
        (psf.alphaBar * exp_term / (2 * pi))[1]
    end

    [ sum([ get_psf_value(psf, float(row), float(col)) for psf in psf_array ]) for
      row in rows, col in cols ]
end


end
