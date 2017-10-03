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
    render_psf(psf, dims)

Render a Celeste PSF on a grid of size `dims`. The PSF is centered in the
grid, with center coordinates `(dims[1]+1) / 2, (dims[2]+1) / 2`.
"""
function render_psf(psf::Array{PsfComponent}, dims::Tuple{Int, Int})
    center = ((dims[1]+1) / 2, (dims[2]+1) / 2)
    stamp = zeros(dims)
    x = zeros(2)
    for pc in psf
        bvn = MultivariateNormal(convert(Array, pc.xiBar),
                                 convert(Array, pc.tauBar))
        for j in 1:dims[2], i in 1:dims[1]
            x[1] = i - center[1]
            x[2] = j - center[2]
            stamp[i, j] += pc.alphaBar * pdf(bvn, x)
        end
    end
    return stamp
end


"""
    AbstractPSFMap

Subtypes are callables that return a PSF stamp given pixel position.
"""
abstract type AbstractPSFMap end


"""
    ConstantPSFMap <: AbstractPSFMap

A non-variable PSF map: calling an instance returns the same PSF stamp
regardless of arguments.
"""
struct ConstantPSFMap <: AbstractPSFMap
    stamp::Matrix{Float64}
end
(psfmap::ConstantPSFMap)(x, y) = copy(psfmap.stamp)
