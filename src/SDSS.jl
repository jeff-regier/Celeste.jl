module SDSS

using CelesteTypes

# FITSIO v0.6.0 removed some of the lower-level functions used here.
# In particular, I think wcslib may still need this.
# Include this as a temporary fix.
using FITSIO.Libcfitsio

using FITSIO
using WCSLIB
using DataFrames

# Just dealing with SDSS stuff.  Now tested.

const bands = ['u', 'g', 'r', 'i', 'z']

function load_stamp_blob(stamp_dir, stamp_id)
    function fetch_image(b)
        band_letter = bands[b]
        filename = "$stamp_dir/stamp-$band_letter-$stamp_id.fits"

        fits = FITS(filename)
        hdr = read_header(fits[1])
        original_pixels = read(fits[1])
        close(fits)
        dn = original_pixels / hdr["CALIB"] + hdr["SKY"]
        nelec = float(int(dn * hdr["GAIN"]))

        # TODO: Does the new FITS file format allow this to be done at a higher level?
        fits_file = FITSIO.Libcfitsio.fits_open_file(filename)
        header_str = FITSIO.Libcfitsio.fits_hdr2str(fits_file)
        FITSIO.Libcfitsio.fits_close_file(fits_file)
        ((wcs,),nrejected) = wcspih(header_str)

        alphaBar = [hdr["PSF_P0"], hdr["PSF_P1"], hdr["PSF_P2"]]
        xiBar = [
            [hdr["PSF_P3"]  hdr["PSF_P4"]],
            [hdr["PSF_P5"]  hdr["PSF_P6"]],
            [hdr["PSF_P7"]  hdr["PSF_P8"]]
        ]'
        tauBar = Array(Float64, 2, 2, 3)
        tauBar[:,:,1] = [[hdr["PSF_P9"] hdr["PSF_P11"]],
                [hdr["PSF_P11"] hdr["PSF_P10"]]]
        tauBar[:,:,2] = [[hdr["PSF_P12"] hdr["PSF_P14"]],
                [hdr["PSF_P14"] hdr["PSF_P13"]]]
        tauBar[:,:,3] = [[hdr["PSF_P15"] hdr["PSF_P17"]],
                [hdr["PSF_P17"] hdr["PSF_P16"]]]

        psf = [PsfComponent(alphaBar[k], xiBar[:, k], tauBar[:, :, k]) for k in 1:3]

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


function load_stamp_catalog_df(cat_dir, stamp_id, blob; match_blob=false)
    # TODO: where is this file format documented?
    cat_fits = FITS("$cat_dir/cat-$stamp_id.fits")
    num_cols = read_key(cat_fits[2], "TFIELDS")[1]
    ttypes = [read_key(cat_fits[2], "TTYPE$i")[1] for i in 1:num_cols]

    df = DataFrame()
    for i in 1:num_cols
        tmp_data = read(cat_fits[2], ttypes[i])        
        df[symbol(ttypes[i])] = tmp_data
    end

    close(cat_fits)

    if match_blob
        camcol_matches = df[:camcol] .== blob[3].camcol_num
        run_matches = df[:run] .== blob[3].run_num
        field_matches = df[:field] .== blob[3].field_num
        df = df[camcol_matches & run_matches & field_matches, :]
    end

    df
end


function load_stamp_catalog(cat_dir, stamp_id, blob; match_blob=false)
    df = load_stamp_catalog_df(cat_dir, stamp_id, blob, match_blob=match_blob)

    function row_to_ce(row)
        x_y = wcss2p(blob[1].wcs, [row[1, :ra], row[1, :dec]]'')[:]

        star_fluxes = zeros(5)
        gal_fluxes = zeros(5)
        fracs_dev = [row[1, :frac_dev], 1 - row[1, :frac_dev]]
        for b in 1:5
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

        if !match_blob  # horrible hack
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


function load_field(field_dir, run_num, camcol_num, frame_num)

    # First, read in the photofield information (the background sky,
    # gain, variance, and calibration).
    photofield_fits = "$field_dir/photoField-$run_num-$camcol_num.fits"
    photofield_fits = FITS(pf_filename)
    @assert length(pf_fits) == 2

    band_gain = read(photofield_fits[2], "gain");
    field_row = read(photofield_fits[2], "field") .== int(frame_num);
    band_dark_variance = collect(read(photofield_fits[2], "dark_variance")[:, field_row]);

    close(pf_fits)

    function fetch_image(b)
        b_letter = bands[b]

        img_filename = "$field_dir/frame-$b_letter-$run_num-$camcol_num-$frame_num.fits"
        img_fits = FITS(img_filename)
        @assert length(img_fits) == 4

        # This is the sky-subtracted and calibrated image.  There are no fields in the first header.
        processed_image = read(img_fits[1]);

        # Read in the sky background.
        sky_image_raw = read(img_fits[3], "ALLSKY");
        sky_x = collect(read(img_fits[3], "XINTERP"));
        sky_y = collect(read(img_fits[3], "YINTERP"));

        # Interpolate to the full image.  Combining the example from
        # http://data.sdss3.org/datamodel/files/BOSS_PHOTOOBJ/frames/RERUN/RUN/CAMCOL/frame.html
        # ...with the documentation from the IDL language:
        # http://www.exelisvis.com/docs/INTERPOLATE.html
        # ...we can see that we are supposed to interpret a point (sky_x[i], sky_y[j])
        # with associated row and column (if, jf) = (floor(sky_x[i]), floor(sky_y[j]))
        # as lying in the square spanned by the points
        # (sky_image_raw[if, jf], sky_image_raw[if + 1, jf + 1]).
        # ...keeping in mind that IDL uses zero indexing:
        # http://www.exelisvis.com/docs/Manipulating_Arrays.html
        sky_grid_vals = ((1:1.:size(sky_image_raw)[1]) - 1, (1:1.:size(sky_image_raw)[2]) - 1);
        sky_grid = CoordInterpGrid(sky_grid_vals, sky_image_raw[:,:,1], BCnearest, InterpLinear);
        sky_image = [ sky_grid[x, y] for x in sky_x, y in sky_y ];

        # This is the calibration vector:
        calib_row = read(img_fits[2]);
        calib_image = [ calib_row[x] for x in 1:size(processed_image)[1], y in 1:size(processed_image)[2] ];

        # Convert to raw electron counts.
        dn = convert(Array{Float64, 2}, (processed_image ./ calib_image .+ sky_image));
        nelec = band_gain[b] * dn;

        img_fits_raw = fits_open_file(img_filename)
        header_str = fits_hdr2str(img_fits_raw)
        close(img_fits_raw)
        ((wcs,),nrejected) = wcspih(header_str)

        # TODO: get the PSF components.
        #psf = [PsfComponent(alphaBar[k], xiBar[:, k], tauBar[:, :, k]) for k in 1:3]

        H, W = size(processed_image)

        # TODO: these need to be stored in the Image object in a sensible way.
        iota = band_gain[b] ./ calib_image
        epsilon = sky_image .* calib_image
        Image(H, W, nelec, b, wcs, epsilon, iota, psf)
    end

    blob = map(fetch_image, 1:5)
end


function mask_image!(mask_img, field_dir, run_num, camcol_num, frame_num)
    # Set the pixels in masK_img to NaN in the places specified bythe fpM file.

    # These are the masking planes used by Dustin's code.    
    # From sdss/dr8.py:
    #   for plane in [ 'INTERP', 'SATUR', 'CR', 'GHOST' ]:
    #            fpM.setMaskedPixels(plane, invvar, 0, roi=roi)
    # TODO: What is the meaning of these?
    const mask_planes = Set({"S_MASK_INTERP", "S_MASK_SATUR", "S_MASK_CR", "S_MASK_GHOST"});

    # http://data.sdss3.org/datamodel/files/PHOTO_REDUX/RERUN/RUN/objcs/CAMCOL/fpM.html
    fpm_filename = "$field_dir/fpM-$run_num-r$camcol_num-$frame_num.fit";
    fpm_fits = FITS(fpm_filename);

    # The last header contains the mask.
    fpm_mask = fpm_fits[12]
    fpm_hdu_indices = read(fpm_mask, "value")

    # Only these rows contain masks.
    masktype_rows = find(read(fpm_mask, "defName") .== "S_MASKTYPE")

    # Apparently attributeName lists the meanings of the HDUs in order.
    mask_types = read(fpm_mask, "attributeName")
    plane_rows = findin(mask_types[masktype_rows], mask_planes)

    for fpm_i in plane_rows
        # You want the HDU in 2 + fpm_mask.value[i] for i in keep_rows (in a 1-indexed language).
        println("Mask type ", mask_types[fpm_i])
        mask_index = 2 + fpm_hdu_indices[fpm_i]
        cmin = read(fpm_fits[mask_index], "cmin")
        cmax = read(fpm_fits[mask_index], "cmax")
        rmin = read(fpm_fits[mask_index], "rmin")
        rmax = read(fpm_fits[mask_index], "rmax")
        row0 = read(fpm_fits[mask_index], "row0")
        col0 = read(fpm_fits[mask_index], "col0")

        @assert all(col0 .== 0)
        @assert all(row0 .== 0)
        @assert length(rmin) == length(cmin) == length(rmax) == length(cmax)

        for block in 1:length(rmin)
            # The ranges are for a 0-indexed language.
            println(block, ": Size of deleted block: ", (cmax[block] + 1 - cmin[block]) * (rmax[block] + 1 - rmin[block]))
            @assert cmax[block] + 1 <= size(mask_img)[1]
            @assert cmin[block] + 1 >= 1
            @assert rmax[block] + 1 <= size(mask_img)[2]
            @assert rmin[block] + 1 >= 1

            # For some reason, the sizes are inconsistent if the rows are read first.
            # What is the mapping from (row, col) to (x, y)?
            # TODO: I think I am reading in the images the wrong way around.
            mask_img[(cmin[block]:cmax[block]) + 1, (rmin[block]:rmax[block]) + 1] = NaN
        end
    end

end

end

