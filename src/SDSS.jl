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

function load_stamp_blob(stamp_dir, stamp_id)
    function fetch_image(b)
        band_letter = ['u', 'g', 'r', 'i', 'z'][b]
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


end

