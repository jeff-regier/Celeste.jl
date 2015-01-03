#!/usr/bin/env julia

module StampBlob

using CelesteTypes

using FITSIO
using WCSLIB


function load_stamp_blob(stamp_dir, stamp_id)
    function fetch_image(b)
        band_letter = ['z', 'i', 'r', 'g', 'u'][b]
        filename = "$stamp_dir/stamp-$band_letter-$stamp_id.fits"

        fits = FITS(filename)
        original_pixels = read(fits[1])
        hdr = readheader(fits[1])
		close(fits)
        dn = original_pixels / hdr["CALIB"] + hdr["SKY"]
        nelec = float(int(dn * hdr["GAIN"]))

        fits_file = fits_open_file(filename)
        header_str = fits_hdr2str(fits_file)
		close(fits_file)
        ((wcs,),nrejected) = wcspih(header_str)

		alphaBar = [hdr["PSF_P0"], hdr["PSF_P1"], hdr["PSF_P2"]]
		xiBar = [
			[hdr["PSF_P3"]  hdr["PSF_P4"]],
			[hdr["PSF_P5"]  hdr["PSF_P6"]],
			[hdr["PSF_P7"]  hdr["PSF_P8"]]
		]'
		SigmaBar = Array(Float64, 2, 2, 3)
		SigmaBar[:, :, 1] = [[hdr["PSF_P9"] hdr["PSF_P11"]], [hdr["PSF_P11"] hdr["PSF_P10"]]]
		SigmaBar[:, :, 2] = [[hdr["PSF_P12"] hdr["PSF_P14"]], [hdr["PSF_P14"] hdr["PSF_P13"]]]
		SigmaBar[:, :, 3] = [[hdr["PSF_P15"] hdr["PSF_P17"]], [hdr["PSF_P17"] hdr["PSF_P16"]]]

		psf = [PsfComponent(alphaBar[k], xiBar[:, k], SigmaBar[:, :, k]) for k in 1:3]

        H, W = size(original_pixels)
		epsilon = median(nelec)
        Image(H, W, nelec, b, wcs, epsilon, psf)
    end

    blob = map(fetch_image, 1:5)
end

end


