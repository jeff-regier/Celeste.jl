module SDSS

using CelesteTypes

using FITSIO
using WCSLIB


function load_stamp_blob(stamp_dir, stamp_id)
    function fetch_image(b)
        band_letter = ['u', 'g', 'r', 'i', 'z'][b]
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
		SigmaBar[:,:,1] = [[hdr["PSF_P9"] hdr["PSF_P11"]], [hdr["PSF_P11"] hdr["PSF_P10"]]]
		SigmaBar[:,:,2] = [[hdr["PSF_P12"] hdr["PSF_P14"]], [hdr["PSF_P14"] hdr["PSF_P13"]]]
		SigmaBar[:,:,3] = [[hdr["PSF_P15"] hdr["PSF_P17"]], [hdr["PSF_P17"] hdr["PSF_P16"]]]

		psf = [PsfComponent(alphaBar[k], xiBar[:, k], SigmaBar[:, :, k]) for k in 1:3]

        H, W = size(original_pixels)
		iota = hdr["GAIN"] / hdr["CALIB"]
		epsilon = hdr["SKY"] * hdr["CALIB"]
        Image(H, W, nelec, b, wcs, epsilon, iota, psf)
    end

    blob = map(fetch_image, 1:5)
end


function load_stamp_catalog(cat_dir, stamp_id, blob)
    cat_fits = fits_open_table("$cat_dir/cat-$stamp_id.fits")
    num_rows = int(fits_read_keyword(cat_fits, "NAXIS2")[1])
    num_cols = int(fits_read_keyword(cat_fits, "TFIELDS")[1])
	ttypes = [rstrip(fits_read_keyword(cat_fits, "TTYPE$i")[1][2:end-1]) for i in 1:num_cols]
	tforms = [rstrip(fits_read_keyword(cat_fits, "TFORM$i")[1][2:end-1]) for i in 1:num_cols]
	col_types = [l in ("D", "E") ? Float64 : l in ("L",) ? Bool : l in ("B", "I") ? Int64 : None for l in tforms]
    table = Array(Any, num_rows, num_cols)
    for i in 1:num_cols
		tmp_data = Array(col_types[i], num_rows)
        fits_read_col(cat_fits, col_types[i], i, 1, 1, tmp_data)
        table[:, i] = tmp_data
    end
    fits_close_file(cat_fits)

	ra_i = findfirst(ttypes, "ra")
	dec_i = findfirst(ttypes, "dec")
	is_star_i = findfirst(ttypes, "is_star")
	b_letter = ['u', 'g', 'r', 'i', 'z']
	fluxes_i = Int64[findfirst(ttypes, "psfflux_$b") for b in b_letter]
    function row_to_cs(row)
		x_y = wcss2p(blob[1].wcs, [row[ra_i], row[dec_i]]'')[:]
		fluxes = row[fluxes_i]
		row[is_star_i] ?
			CatalogStar(x_y, fluxes) :
			CatalogGalaxy(x_y, fluxes, 42., [42., 0., 0])
    end
    CatalogEntry[row_to_cs(table[i, :][:]) for i in 1:num_rows]
end


function load_field(field_dir, run_num, camcol_num, frame_num)
	pf_filename = "$field_dir/photoField-$run_num-$camcol_num.fits"
	pf_fits_raw = fits_open_table(pf_filename)

	num_rows = int(fits_read_keyword(pf_fits_raw, "NAXIS2")[1])
	pf_frames = Array(Int64, num_rows)
	fits_read_col(pf_fits_raw, Int64, 6, 1, 1, pf_frames) # 6 = field/frame num
	pf_rownum = searchsorted(pf_frames, int(frame_num))[1]

	# this doesn't look right...way too small
	pf_sky = Array(Float64, 5) #1 per band
    fits_read_col(pf_fits_raw, Float64, 70, pf_rownum, 1, pf_sky) # 70 = sky

	pf_gain = Array(Float64, 5) #1 per band
    fits_read_col(pf_fits_raw, Float64, 94, pf_rownum, 1, pf_gain) # 94 = gain

	close(pf_fits_raw)

    function fetch_image(b)
        b_letter = ['u', 'g', 'r', 'i', 'z'][b]
        img_filename = "$field_dir/frame-$b_letter-$run_num-$camcol_num-$frame_num.fits"

        img_fits = FITS(img_filename)
        original_pixels = read(img_fits[1])
        img_hdr1 = readheader(img_fits[1])
        img_hdr2 = readheader(img_fits[2])
		close(img_fits)

        img_fits_raw = fits_open_file(img_filename)
        header_str = fits_hdr2str(img_fits_raw)
		close(img_fits_raw)
        ((wcs,),nrejected) = wcspih(header_str)

		img_tbl = fits_open_table(img_filename)
		fits_movabs_hdu(img_tbl, 3)
		allsky = Array(Float64, 5)
		fits_read_col(img_tbl, Float64, 1, 1, 1, allsky)
		close(img_tbl)

		# temporary, until dustin gets back to me
		package_dat = joinpath(Pkg.dir("Celeste"), "dat")
		stamp_id = "164.4311-39.0359"
        filename = "$package_dat/stamp-$b_letter-$stamp_id.fits"
        fits = FITS(filename)
        hdr = readheader(fits[1])
        close(fits)

        dn = original_pixels / img_hdr1["NMGY"] + median(allsky)
        nelec = float(int(dn * pf_gain[b]))

		alphaBar = [hdr["PSF_P0"], hdr["PSF_P1"], hdr["PSF_P2"]]
		xiBar = [
			[hdr["PSF_P3"]  hdr["PSF_P4"]],
			[hdr["PSF_P5"]  hdr["PSF_P6"]],
			[hdr["PSF_P7"]  hdr["PSF_P8"]]
		]'
		SigmaBar = Array(Float64, 2, 2, 3)
		SigmaBar[:,:,1] = [[hdr["PSF_P9"] hdr["PSF_P11"]], [hdr["PSF_P11"] hdr["PSF_P10"]]]
		SigmaBar[:,:,2] = [[hdr["PSF_P12"] hdr["PSF_P14"]], [hdr["PSF_P14"] hdr["PSF_P13"]]]
		SigmaBar[:,:,3] = [[hdr["PSF_P15"] hdr["PSF_P17"]], [hdr["PSF_P17"] hdr["PSF_P16"]]]

		psf = [PsfComponent(alphaBar[k], xiBar[:, k], SigmaBar[:, :, k]) for k in 1:3]

        H, W = size(original_pixels)
		iota = hdr["GAIN"] / hdr["CALIB"]
		epsilon = hdr["SKY"] * hdr["CALIB"]
        Image(H, W, nelec, b, wcs, epsilon, iota, psf)
    end

    blob = map(fetch_image, 1:5)
end


function load_catalog(field_dir, run_num, camcol_num, frame_num)
    function common_catalog(cat_str)
        @assert cat_str == "star" || cat_str == "gal"
        cat_file = "$field_dir/calibObj-$run_num-$camcol_num-$cat_str.fits"
        cat_fits = fits_open_table(cat_file)

        num_rows = int(fits_read_keyword(cat_fits, "NAXIS2")[1])
        row_frames = Array(Int64, num_rows)
        fits_read_col(cat_fits, Int64, 4, 1, 1, row_frames)  # 4 = field/frame num
        frame_range = searchsorted(row_frames, int(frame_num))
        num_frame_rows = length(frame_range)

        ras = Array(Float64, num_frame_rows)
        decs = Array(Float64, num_frame_rows)
        fits_read_col(cat_fits, Float64, 21, frame_range[1], 1, ras)  # 21 = RA
        fits_read_col(cat_fits, Float64, 22, frame_range[1], 1, decs)  # 22 = DEC

        cat_fits, frame_range, ras, decs
    end

    star_fits, star_range, star_ras, star_decs = common_catalog("star")
    star_cat = [CatalogStar([star_ras[i], star_decs[i]], ones(5) * 50000)
                    for i in 1:length(star_range)]

    gal_fits, gal_range, gal_ras, gal_decs = common_catalog("gal")
    gal_cat = [CatalogGalaxy([gal_ras[i], gal_decs[i]],   
                ones(5) * 50000, 0.5, [2., 0., 2.])
                    for i in 1:length(gal_range)]
    vcat(star_cat, gal_cat)
end


end

