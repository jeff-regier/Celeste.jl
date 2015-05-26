module SDSS

VERSION < v"0.4.0-dev" && using Docile
using CelesteTypes
using FITSIO
using WCSLIB
using DataFrames
using Grid
using GaussianMixtures

# FITSIO v0.6.0 removed some of the lower-level functions used here.
# In particular, I think wcslib may still need this.
# Include this as a temporary fix.
using FITSIO.Libcfitsio


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


@doc """
Load data from a psField file, which contains the point spread function.

Args:
 - field_dir: The directory of the file
 - run_num: The run number
 - camcol_num: The camcol number
 - frame_num: The frame number
 - b: The filter band (a number from 1 to 5)

Returns:
 - rrows: A matrix of flattened eigenimages.
 - rnrow: The number of rows in an eigenimage.
 - rncol: The number of columns in an eigenimage.
 - cmat: The coefficients of the weight polynomial (see below).

The point spread function is represented as an rnrow x rncol image
showing what a true point source would look as viewed through the optics.
This image varies across the field, and to parameterize this variation,
the PSF is represented as a linear combination of four "eigenimages",
with weights that vary across the image.  See get_psf_at_point()
for more details.
""" ->
function load_psf_data(field_dir, run_num, camcol_num, frame_num, b)
    @assert 1 <= b <= 5
    psf_filename = "$field_dir/psField-$run_num-$camcol_num-$frame_num.fit"
    psf_fits = FITS(psf_filename)
    psf_hdu = psf_fits[b + 1]

    nrows = read_key(psf_hdu, "NAXIS2")[1]
    nrow_b = read(psf_hdu, "nrow_b")[1]
    ncol_b = read(psf_hdu, "ncol_b")[1]
    rnrow = read(psf_hdu, "rnrow")[1]
    rncol = read(psf_hdu, "rncol")[1]
    cmat = convert(Array{Float64, 3}, read(psf_hdu, "c"))

    # Some low-level FITSIO rigamarole until the high-level interface can
    # read variable length binary rows.
    
    # Move to the appropriate header.
    FITSIO.Libcfitsio.fits_movabs_hdu(psf_hdu.fitsfile, psf_hdu.ext)
    rrows_colnum = FITSIO.Libcfitsio.fits_get_colnum(psf_hdu.fitsfile, "RROWS")
    rrows = zeros(Float64, rnrow * rncol, nrows)
    for rownum in 1:nrows
        repeat, offset = FITSIO.Libcfitsio.fits_read_descriptll(psf_hdu.fitsfile, rrows_colnum, rownum)
        @assert repeat[1] == rnrow * rncol
        result = zeros(repeat[1])
        FITSIO.Libcfitsio.fits_read_col(psf_hdu.fitsfile, rrows_colnum, rownum, 1, result)
        rrows[:, rownum] = result
    end
    close(psf_fits)

    # Only the first (nrow_b, ncol_b) submatrix of cmat is used for reasons obscure
    # to the author.
    rrows, rnrow, rncol, cmat[1:nrow_b, 1:ncol_b, :]
end

@doc """
Evaluate a gmm object at the data points x_mat.
""" ->
function evaluate_gmm(gmm::GMM, x_mat::Array{Float64, 2})
    post = gmmposterior(gmm, x_mat) 
    exp(post[2]) * gmm.w;
end


@doc """
Fit a mixture of 2d Gaussians to a PSF image (evaluated at a single point).

Args:
 - psf: An matrix image of the point spread function, e.g. as returned by get_psf_at_point.
 - tol: The target mean sum of squared errors between the MVN fit and the raw psf.
 - max_iter: The maximum number of EM steps to take.

 Returns:
  - A GMM object containing the fit

 Use an EM algorithm to fit a mixture of three multivariate normals to the
 point spread function.  Although the psf is continuous-valued, we use EM by
 fitting as if we had gotten psf[x, y] data points at the image location [x, y].
 As of writing weighted data matrices of this form were not supported in GaussianMixtures.

 Naturally, the mixture only matches the psf up to scale.
""" ->
function fit_psf_gaussians(psf::Array{Float64, 2}; tol = 1e-9, max_iter = 500)

    function sigma_for_gmm(sigma_mat)
        # Returns a matrix suitable for storage in the field gmm.Σ
        Triangular(GaussianMixtures.cholinv(sigma_mat), :U, false)
    end

    # TODO: Is it ok that it is coming up negative in some points?
    if (any(psf .< 0))
        warn("Some psf values are negative.")
        psf[ psf .< 0 ] = 0
    end

    # Data points at which the psf is evaluated in matrix form.
    x_prod = [ Float64[i, j] for i=1:size(psf, 1), j=1:size(psf, 2) ]
    x_mat = Float64[ x_row[col] for x_row=x_prod, col=1:2 ]

    # Unscale the fit vector so we can compare the sum of squared differences
    # between the psf and our mixture of normals as a convergence criterion.
    psf_scale = sum(psf)
    psf_mat = Float64[ psf[x_row[1], x_row[2]] / psf_scale for x_row=x_prod ];

    # Use the GaussianMixtures package to evaluate our likelihoods.  In order to
    # avoid automatically choosing an initialization point, hard-code three
    # gaussians for now.  Ignore their initialization, which would not be
    # weighted by the psf.
    gmm = GMM(3, x_mat; kind=:full, nInit=0)

    # Get the scale for the starting point from the whole image.
    psf_center = Float64[ (size(psf, i) - 1) / 2 for i=1:2 ]
    x_centered = broadcast(-, x_mat, psf_center')
    psf_starting_var = x_centered' * (x_centered .* psf_mat)

    # Hard-coded initialization.
    gmm.μ[1, :] = psf_center
    gmm.Σ[1] = sigma_for_gmm(psf_starting_var)

    gmm.μ[2, :] = psf_center - Float64[2, 2]
    gmm.Σ[2] = sigma_for_gmm(psf_starting_var)

    gmm.μ[3, :] = psf_center + Float64[2, 2]
    gmm.Σ[3] = sigma_for_gmm(psf_starting_var)

    gmm.w = ones(gmm.n) / gmm.n

    iter = 1
    err_diff = Inf
    last_err = Inf
    fit_done = false

    # post contains the posterior information about the values of the
    # mixture as well as the probabilities of each component.
    post = gmmposterior(gmm, x_mat) 

    while !fit_done
        # Update gmm using last value of post.  post[1] contains
        # posterior probabilities of the indicators.
        z = post[1] .* psf_mat
        z_sum = collect(sum(z, 1))
        new_w = z_sum / sum(z_sum)
        gmm.w = new_w
        for d=1:gmm.n
            if new_w[d] > 1e-6
                new_mean = sum(x_mat .* z[:, d], 1) / z_sum[d]
                x_centered = broadcast(-, x_mat, new_mean)
                x_cov = x_centered' * (x_centered .* z[:, d]) / z_sum[d]

                gmm.μ[d, :] = new_mean
                gmm.Σ[d] = sigma_for_gmm(x_cov)
            else
                warn("Component $d has very small probability.")
            end
        end

        # Get next posterior and check for convergence.  post[2] contains
        # the log densities at each point.
        post = gmmposterior(gmm, x_mat) 
        gmm_fit = exp(post[2]) * gmm.w;
        err = mean((gmm_fit - psf_mat) .^ 2)
        err_diff = last_err - err
        last_err = err
        if isnan(err)
            error("NaN in MVN PSF fit.")
        end

        iter = iter + 1
        if err_diff < tol
            fit_done = true
        elseif iter >= max_iter
            warn("PSF MVN fit: max_iter exceeded")
            fit_done = true
        end

        println("Fitting psf: $iter: $err_diff")
    end

    gmm
end


@doc """
Convert a GaussianMixtures GMM object to an array of Celect PsfComponents.

Args:
 - gmm: A GaussianMixtures GMM object (e.g. as returned by fit_psf_gaussians)

 Returns:
  - An array of PsfComponent objects.
""" ->
function convert_gmm_to_celeste(gmm::GMM)
    function convert_gmm_component_to_celeste(gmm::GMM, d)
        PsfComponent(gmm.w[d], collect(means(gmm)[d, :]), covars(gmm)[d])
    end

    [ convert_gmm_component_to_celeste(gmm, d) for d=1:gmm.n ]
end


@doc """
Using data from a psField file, evaluate the PSF for a source at given point.

Args:
 - row: The row of the point source (may be a float).
 - col: The column of the point source (may be a float).
 - rrows: An (rnrow * rncol) by (k) matrix of flattened eigenimages.
 - rnrow: The number of rows in the eigenimage.
 - rncol: The number of columns in the eigenimage.
 - cmat: An (:, :, k)-array of coefficients of the weight polynomial.

Returns:
 - An rnrow x rncol image of the PSF at (row, col)

The PSF is represented as a weighted combination of "eigenimages" (stored
in rrows), where the weights vary smoothly across points (row, col) in the image
as a polynomial of the form
weight[k](row, col) = sum_{i,j} cmat[i, j, k] * (rcs * row) ^ i (rcs * col) ^ j

This function is based on the function getPsfAtPoints in astrometry.net:
https://github.com/dstndstn/astrometry.net/blob/master/sdss/common.py#L953
""" ->
function get_psf_at_point(row::Float64, col::Float64,
                          rrows::Array{Float64, 2}, rnrow::Int32, rncol::Int32, 
                          cmat::Array{Float64, 3})

    # This is a coordinate transform to keep the polynomial coefficients
    # to a reasonable size.
    const rcs = 0.001

    # rrows' image data is in the first column a flattened form.
    # The second dimension is the number of eigen images, which should
    # match the number of coefficient arrays.
    k = size(rrows)[2]
    @assert k == size(cmat)[3]

    nrow_b = size(cmat)[1]
    ncol_b = size(cmat)[2]

    # Get the weights.
    coeffs_mat = [ (row * rcs) ^ i * (col * rcs) ^ j for i=0:(nrow_b - 1), j=0:(ncol_b - 1)]
    weight_mat = zeros(nrow_b, ncol_b)
    for k = 1:3, i = 1:nrow_b, j = 1:ncol_b
        weight_mat[i, j] += cmat[i, j, k] * coeffs_mat[i, j]
    end

    # Weight the images in rrows and reshape them into matrix form.
    # It seems I need to convert for reshape to work.  :(
    psf = reshape(reduce(sum, [ rrows[:, i] * weight_mat[i] for i=1:k]),
                  (convert(Int64, rnrow), convert(Int64, rncol)))

    psf
end


@doc """
Load relevant data from a photoField file.

Args:
 - field_dir: The directory of the file
 - run_num: The run number
 - camcol_num: The camcol number
 - frame_num: The frame number
 - b: The filter band (a number from 1 to 5)

Returns:
 - band_gain: An array of gains for the five bands
 - band_dark_variance: An array of dark variances for the five bands
""" ->
function load_photo_field(field_dir, run_num, camcol_num, frame_num)
    # First, read in the photofield information (the background sky,
    # gain, variance, and calibration).
    pf_filename = "$field_dir/photoField-$run_num-$camcol_num.fits"
    pf_fits = FITS(pf_filename)
    @assert length(pf_fits) == 2

    field_row = read(pf_fits[2], "field") .== int(frame_num)
    band_gain = collect(read(pf_fits[2], "gain")[:, field_row])
    band_dark_variance = collect(read(pf_fits[2], "dark_variance")[:, field_row])

    close(pf_fits)

    band_gain, band_dark_variance
end


@doc """
Load the raw electron counts, calibration vector, and sky background from a field.

Args:
 - field_dir: The directory of the file
 - run_num: The run number
 - camcol_num: The camcol number
 - frame_num: The frame number
 - b: The filter band (a number from 1 to 5)
 - gain: The gain for this band (e.g. as read from photoField)

Returns:
 - nelec: An image of raw electron counts in nanomaggies
 - calib_col: A column of calibration values (the same for every column of the image) 
 - sky_grid: A CoordInterpGrid bilinear interpolation object

The meaing of the frame data structures is thoroughly documented here:
http://data.sdss3.org/datamodel/files/BOSS_PHOTOOBJ/frames/RERUN/RUN/CAMCOL/frame.html
""" ->
function load_raw_field(field_dir, run_num, camcol_num, frame_num, b, gain)
    @assert 1 <= b <= 5
    b_letter = bands[b]

    img_filename = "$field_dir/frame-$b_letter-$run_num-$camcol_num-$frame_num.fits"
    img_fits = FITS(img_filename)
    @assert length(img_fits) == 4

    # This is the sky-subtracted and calibrated image.
    processed_image = read(img_fits[1])

    # Read in the sky background.
    sky_image_raw = read(img_fits[3], "ALLSKY")
    sky_x = collect(read(img_fits[3], "XINTERP"))
    sky_y = collect(read(img_fits[3], "YINTERP"))

    # Interpolate the sky to the full image.  Combining the example from
    # http://data.sdss3.org/datamodel/files/BOSS_PHOTOOBJ/frames/RERUN/RUN/CAMCOL/frame.html
    # ...with the documentation from the IDL language:
    # http://www.exelisvis.com/docs/INTERPOLATE.html
    # ...we can see that we are supposed to interpret a point (sky_x[i], sky_y[j])
    # with associated row and column (if, jf) = (floor(sky_x[i]), floor(sky_y[j]))
    # as lying in the square spanned by the points
    # (sky_image_raw[if, jf], sky_image_raw[if + 1, jf + 1]).
    # ...keeping in mind that IDL uses zero indexing:
    # http://www.exelisvis.com/docs/Manipulating_Arrays.html
    sky_grid_vals = ((1:1.:size(sky_image_raw)[1]) - 1, (1:1.:size(sky_image_raw)[2]) - 1)
    sky_grid = CoordInterpGrid(sky_grid_vals, sky_image_raw[:,:,1], BCnearest, InterpLinear)
    sky_image = [ sky_grid[x, y] for x in sky_x, y in sky_y ]

    # This is the calibration vector:
    calib_col = read(img_fits[2])
    calib_image = [ calib_col[row] for
                    row in 1:size(processed_image)[1],
                    col in 1:size(processed_image)[2] ]

    # Convert to raw electron counts.  Note that these may not be close to integers
    # due to the analog to digital conversion process in the telescope.
    nelec = gain * convert(Array{Float64, 2}, (processed_image ./ calib_image .+ sky_image))

    nelec, calib_col, sky_grid
end

# img_fits_raw = fits_open_file(img_filename)
# header_str = fits_hdr2str(img_fits_raw)
# close(img_fits_raw)
# ((wcs,),nrejected) = wcspih(header_str)


@doc """
Set the pixels in mask_img to NaN in the places specified by the fpM file.

Args:
 - mask_img: The image to be masked (updated in place)
 - field_dir: The directory of the file
 - run_num: The run number
 - camcol_num: The camcol number
 - frame_num: The frame number

Returns:
 - Updates mask_img in place by setting to NaN all the pixels specified.

 This is based on the function setMaskedPixels in astrometry.net:
 https://github.com/dstndstn/astrometry.net/
""" ->
function mask_image!(mask_img, field_dir, run_num, camcol_num, frame_num, band;
                     python_indexing = false,
                     mask_planes = Set({"S_MASK_INTERP", "S_MASK_SATUR", "S_MASK_CR", "S_MASK_GHOST"}))
    # The default mask planes are those used by Dustin's astrometry.net code.    
    # See the comments in sdss/dr8.py for fpM.setMaskedPixels
    # and the function sdss/common.py:fpM.setMaskedPixels
    #
    # interp = pixel was bad and interpolated over
    # satur = saturated
    # cr = cosmic ray
    # ghost = artifact from the electronics.

    # http://data.sdss3.org/datamodel/files/PHOTO_REDUX/RERUN/RUN/objcs/CAMCOL/fpM.html
    band_letter = bands[band]
    fpm_filename = "$field_dir/fpM-$run_num-$band_letter$camcol_num-$frame_num.fit"
    fpm_fits = FITS(fpm_filename)

    # The last header contains the mask.
    fpm_mask = fpm_fits[12]
    fpm_hdu_indices = read(fpm_mask, "value")

    # Only these rows contain masks.
    masktype_rows = find(read(fpm_mask, "defName") .== "S_MASKTYPE")

    # Apparently attributeName lists the meanings of the HDUs in order.
    mask_types = read(fpm_mask, "attributeName")
    plane_rows = findin(mask_types[masktype_rows], mask_planes)

    # Make sure each mask is present.  Is this check appropriate for all mask files?
    @assert length(plane_rows) == length(mask_planes)

    for fpm_i in plane_rows
        # You want the HDU in 2 + fpm_mask.value[i] for i in keep_rows (in a 1-indexed language).
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
            @assert cmax[block] + 1 <= size(mask_img)[1]
            @assert cmin[block] + 1 >= 1
            @assert rmax[block] + 1 <= size(mask_img)[2]
            @assert rmin[block] + 1 >= 1

            # Some notes:
            # See astrometry.net//sdss/common.py:SetMaskedPixels, which I currently assume is correct.
            # - In contrast with  julia, the numpy matrix index range [3:5, 3:5] contains four
            #   pixels, not six.  However, if the numpy is correct, then fpM files contain
            #   many bad rows that don't get masked at all since cmin == cmax
            #   or rmin == rmax.  For this reason, I think the python might be erroneous.
            # - For some reason, the sizes are inconsistent if the rows are read first.
            #   I presume that either these names are strange or I am supposed to read
            #   the image from the frame and transpose it.
            # - Julia is 1-indexed, not 0-indexed.

            # Give the option of using Dustin's python indexing or not.
            if python_indexing
                mask_rows = (cmin[block] + 1):(cmax[block])
                mask_cols = (rmin[block] + 1):(rmax[block])
            else
                mask_rows = (cmin[block] + 1):(cmax[block] + 1)
                mask_cols = (rmin[block] + 1):(rmax[block] + 1)
            end

            mask_img[mask_rows, mask_cols] = NaN
        end
    end
end

end

