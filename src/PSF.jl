module PSF

VERSION < v"0.4.0-dev" && using Docile

import CelesteTypes
import GaussianMixtures

@doc """
Evaluate a gmm object at the data points x_mat.
""" ->
function evaluate_gmm(gmm::GaussianMixtures.GMM, x_mat::Array{Float64, 2})
    post = GaussianMixtures.gmmposterior(gmm, x_mat) 
    exp(post[2]) * gmm.w;
end


@doc """
Fit a mixture of 2d Gaussians to a PSF image (evaluated at a single point).

Args:
 - psf: An matrix image of the point spread function, e.g. as returned by get_psf_at_point.
 - tol: The target mean sum of squared errors between the MVN fit and the raw psf.
 - max_iter: The maximum number of EM steps to take.

 Returns:
  - A GaussianMixtures.GMM object containing the fit

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
    gmm = GaussianMixtures.GMM(3, x_mat; kind=:full, nInit=0)

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
    post = GaussianMixtures.gmmposterior(gmm, x_mat) 

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
        post = GaussianMixtures.gmmposterior(gmm, x_mat) 
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
Convert a GaussianMixtures.GMM object to an array of Celect PsfComponents.

Args:
 - gmm: A GaussianMixtures.GMM object (e.g. as returned by fit_psf_gaussians)

 Returns:
  - An array of PsfComponent objects.
""" ->
function convert_gmm_to_celeste(gmm::GaussianMixtures.GMM)
    function convert_gmm_component_to_celeste(gmm::GaussianMixtures.GMM, d)
        CelesteTypes.PsfComponent(gmm.w[d],
            collect(GaussianMixtures.means(gmm)[d, :]), GaussianMixtures.covars(gmm)[d])
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
...where row and col are zero-indexed.

This function is based on the function sdss_psf_at_points in astrometry.net:
https://github.com/dstndstn/astrometry.net/blob/master/util/sdss_psf.py
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
    k_tot = size(rrows)[2]
    @assert k_tot == size(cmat)[3]

    nrow_b = size(cmat)[1]
    ncol_b = size(cmat)[2]

    # Get the weights.  The row and column are intended to be
    # zero-indexed.
    coeffs_mat = [ ((row - 1) * rcs) ^ i * ((col - 1) * rcs) ^ j for
                    i=0:(nrow_b - 1), j=0:(ncol_b - 1)]
    weight_mat = zeros(k_tot)
    for k = 1:k_tot, i = 1:nrow_b, j = 1:ncol_b
        weight_mat[k] += cmat[i, j, k] * coeffs_mat[i, j]
    end

    # Weight the images in rrows and reshape them into matrix form.
    # It seems I need to convert for reshape to work.  :(
    psf = reshape(sum([ rrows[:, i] * weight_mat[i] for i=1:k_tot]),
                  (convert(Int64, rnrow), convert(Int64, rncol)))

    psf
end

end