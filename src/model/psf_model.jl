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
immutable PsfComponent
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

function PsfComponent(alphaBar::Float64, xiBar::Vector{Float64},
                      tauBar::Matrix{Float64})
    PsfComponent(alphaBar, xiBar, tauBar, tauBar^-1, logdet(tauBar))
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
    width_scale * sqrt(eigvals(cov_est)[end]) * alpha_norm
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


