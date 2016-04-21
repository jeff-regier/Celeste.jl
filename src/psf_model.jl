# The number of Gaussian components in the PSF.
const psf_K = 2

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
    xiBar::Vector{Float64}
    tauBar::Matrix{Float64}

    tauBarInv::Matrix{Float64}
    tauBarLd::Float64

    function PsfComponent(alphaBar::Float64, xiBar::Vector{Float64},
                          tauBar::Matrix{Float64})
        new(alphaBar, xiBar, tauBar, tauBar^-1, logdet(tauBar))
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
immutable RawPSF
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


