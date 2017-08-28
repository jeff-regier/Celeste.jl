"""
A single normal component of the point spread function.
All quantities are in pixel coordinates.

Args:
  alphaBar: The scalar weight of the component.
  xiBar: The 2x1 location vector
  tauBar: The 2x2 covariance

Attributes:
  alphaBar: The scalar weight of the component.
  xiBar: The 2x1 location vector
  tauBar: The 2x2 covariance (tau_bar in the ICML paper)
  tauBarInv: The 2x2 precision
  tauBarLd: The log determinant of the covariance
"""
struct PsfComponent
    alphaBar::Float64  # TODO: use underscore
    xiBar::SVector{2,Float64}
    tauBar::SMatrix{2,2,Float64,4}

    tauBarInv::SMatrix{2,2,Float64,4}
    tauBarLd::Float64

    function PsfComponent(alphaBar::Float64, xiBar::SVector{2,Float64},
                          tauBar::SMatrix{2,2,Float64,4})
        new(alphaBar, xiBar, tauBar, inv(tauBar), log(det(tauBar)))
    end
end


"""
SDSS representation of a spatially variable PSF. The PSF is represented as
a weighted combination of eigenimages (stored in `rrows`), where the weights
vary smoothly across the image as a polynomial of the form

```
weight[k](x, y) = sum_{i,j} cmat[i, j, k] * (rcs * x)^i (rcs * y)^j
```

where `rcs` is a coordinate transformation and `x` and `y` are zero-indexed.
"""
struct RawPSF
    rrows::Array{Float64,2}  # A matrix of flattened eigenimages.
    rnrow::Int  # The number of rows in an eigenimage.
    rncol::Int  # The number of columns in an eigenimage.
    cmat::Array{Float64,3}  # The coefficients of the weight polynomial

    function RawPSF(rrows::Array{Float64, 2}, rnrow::Integer, rncol::Integer,
                     cmat::Array{Float64, 3})
        # rrows contains eigen images. Each eigen image is along the first
        # dimension in a flattened form. Check that dimensions match up.
        @assert size(rrows, 1) == rnrow * rncol

        # The second dimension is the number of eigen images, which should
        # match the number of coefficient arrays.
        @assert size(rrows, 2) == size(cmat, 3)

        return new(rrows, Int(rnrow), Int(rncol), cmat)
    end
end


function get_psf_width(psf::Array{PsfComponent}; width_scale=1.0)
    # A heuristic measure of the PSF width based on an anology
    # with it being a mixture of normals.    Note that it is not an actual
    # mixture of normals, and in particular that sum(alphaBar) \ne 1.

    # The PSF is not necessarily centered at (0, 0), but we want a measure
    # of its maximal width around (0, 0), not around its center.
    # Approximate this by finding the covariance of a point randomly drawn
    # from a mixture of gaussians.
    alpha_norm = sum([ psf_comp.alphaBar for psf_comp in psf ])
    cov_est = zeros(Float64, 2, 2)
    for psf_comp in psf
        cov_est +=
            psf_comp.alphaBar * (psf_comp.xiBar * psf_comp.xiBar' + psf_comp.tauBar) /
            alpha_norm
    end

    # Return the twice the sd of the most spread direction, scaled by the total
    # mass in the PSF.
    width_scale * sqrt(eig(cov_est)[1][end]) * alpha_norm
end


"""
psf(x, y)

Evaluate the PSF at the given image coordinates. The size of the result is
will be `(psf.rnrow, psf.rncol)`, with the PSF (presumably) centered in the
stamp.

This function was originally based on the function sdss_psf_at_points
in astrometry.net:
https://github.com/dstndstn/astrometry.net/blob/master/util/sdss_psf.py
"""
function eval_psf(psf::RawPSF, x::Real, y::Real)
    const RCS = 0.001  # A coordinate transform to keep polynomial
                       # coefficients to a reasonable size.
    nk = size(psf.rrows, 2)  # number of eigen images.

    # initialize output stamp
    stamp = zeros(psf.rnrow, psf.rncol)

    # Loop over eigen images
    for k=1:nk
        # calculate the weight for the k-th eigen image from psf.cmat.
        # Note that the image coordinates and coefficients are intended
        # to be zero-indexed.
        w = 0.0
        for j=1:size(psf.cmat, 2), i=1:size(psf.cmat, 1)
            w += (psf.cmat[i, j, k] *
                  (RCS * (x - 1.0))^(i-1) * (RCS * (y - 1.0))^(j-1))
        end

        # add the weighted k-th eigen image to the output stamp
        for i=1:length(stamp)
            stamp[i] += w * psf.rrows[i, k]
        end
    end

    return stamp
end


