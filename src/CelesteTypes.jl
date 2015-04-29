# written by Jeffrey Regier
# jeff [at] stat [dot] berkeley [dot] edu

module CelesteTypes

export CatalogEntry
export band_letters

export Image, Blob, SkyPatch, ImageTile, PsfComponent
export GalaxyComponent, GalaxyPrototype, galaxy_prototypes
export effective_radii

export ModelParams, PriorParams
export VariationalParams, FreeVariationalParams, RectVariationalParams

export SensitiveFloat

export zero_sensitive_float, const_sensitive_param, clear!

export ParamIndex, ids, ids_free, all_params, all_params_free
export star_pos_params, galaxy_pos_params
export D, B, Ia

using Util

import FITSIO
import Distributions
import WCSLIB

const band_letters = ['u', 'g', 'r', 'i', 'z']


# The number of components in the color prior.
const D = 2

# The number of types of celestial objects (here, stars and galaxies).
const Ia = 2

# The number of bands (colors).
const B = 5

type CatalogEntry
    pos::Vector{Float64}
    is_star::Bool
    star_fluxes::Vector{Float64}
    gal_fluxes::Vector{Float64}
    gal_frac_dev::Float64
    gal_ab::Float64
    gal_angle::Float64
    gal_scale::Float64
end

############################################

immutable GalaxyComponent
    # Parameters of a single normal component of a galaxy.
    #
    # Attributes:
    #   etaBar: The weight of the galaxy component
    #   nuBar: The scale of the galaxy component

    etaBar::Float64
    nuBar::Float64
end

typealias GalaxyPrototype Vector{GalaxyComponent}

function get_galaxy_prototypes()
    # Pre-defined shapes for galaxies.
    #
    # Returns:
    #   dev_prototype: An array of GalaxyComponent for de Vaucouleurs galaxy types
    #   exp_prototype: An array of GalaxyComponent for exponenttial galaxy types

    dev_amp = [
        4.26347652e-02, 2.40127183e-01, 6.85907632e-01, 1.51937350e+00,
        2.83627243e+00, 4.46467501e+00, 5.72440830e+00, 5.60989349e+00]
    dev_amp /= sum(dev_amp)
    dev_var = [
        2.23759216e-04, 1.00220099e-03, 4.18731126e-03, 1.69432589e-02,
        6.84850479e-02, 2.87207080e-01, 1.33320254e+00, 8.40215071e+00]

	exp_amp = [
        2.34853813e-03, 3.07995260e-02, 2.23364214e-01,
        1.17949102e+00, 4.33873750e+00, 5.99820770e+00]
    exp_amp /= sum(exp_amp)
    exp_var = [
        1.20078965e-03, 8.84526493e-03, 3.91463084e-02,
        1.39976817e-01, 4.60962500e-01, 1.50159566e+00]

	# Adjustments to the effective radius hard-coded above.
	# (The effective radius is the distance from the center containing half the light.)
	effective_radii = [1.078031, 0.928896]
	dev_var /= effective_radii[1]^2
	exp_var /= effective_radii[2]^2

    exp_prototype = [GalaxyComponent(exp_amp[j], exp_var[j]) for j in 1:6]
    dev_prototype = [GalaxyComponent(dev_amp[j], dev_var[j]) for j in 1:8]
    (dev_prototype, exp_prototype)
end

const galaxy_prototypes = get_galaxy_prototypes()


immutable PsfComponent
    # A single normal component of the point spread function.
    #
    # Args:
    #   alphaBar: The scalar weight of the component. 
    #   xiVar: The 2x1 location vector
    #   Sigmabar: The 2x2 covariance
    #
    # Attributes:
    #   alphaBar: The scalar weight of the component. 
    #   xiVar: The 2x1 location vector
    #   Sigmabar: The 2x2 covariance (tau_bar in the ICLM paper)
    #   tauBarInv: The 2x2 precision
    #   tauBarLd: The log determinant of the covariance

    alphaBar::Float64  # TODO: use underscore
    xiBar::Vector{Float64}
    tauBar::Matrix{Float64}

    tauBarInv::Matrix{Float64}
    tauBarLd::Float64

    PsfComponent(alphaBar::Float64, xiBar::Vector{Float64},
            tauBar::Matrix{Float64}) = begin
        new(alphaBar, xiBar, tauBar, tauBar^-1, logdet(tauBar))
    end
end

type Image
    # An image for a single color.

    # The image height.
    H::Int64

    # The image width.
    W::Int64

    # An HxW matrix of pixel intensities.
    pixels::Matrix{Float64}

    # The band id (takes on values from 1 to 5).
    b::Int64

    # World coordinates
    wcs::WCSLIB.wcsprm

    # The background noise in nanomaggies.
    epsilon::Float64

    # The expected number of photons contributed to this image
    # by a source 1 nanomaggie in brightness.
    iota::Float64

    # The components of the point spread function.
    psf::Vector{PsfComponent}

    # SDSS-specific identifiers. A field is a particular region of the sky.
    # A Camcol is the output of one camera column as part of a Run.
    run_num::Int64
    camcol_num::Int64
    field_num::Int64
end

immutable ImageTile
    # Tiles of pixels that share the same set of    
    # relevant sources (or other calculations).

    # These are in tile coordinates --- not pixel or sky coordinates.
    # (I.e., the range of hh and ww are the number of horizontal
    #  and vertical tiles in the image, respectively.)

    hh::Int64
    ww::Int64
    img::Image
end

# A vector of images, one for each color.
typealias Blob Vector{Image}

immutable SkyPatch #pixel coordinates for now, soon wcs
    # The amount of sky affected by a
    # source (regardless of the tiling).
    center::Vector{Float64}
    radius::Float64
end


#########################################################

immutable PriorParams
    a::Float64  # formerly Phi
    r::Vector{(Float64, Float64)}   # formerly Upsilon, Psi
    k::Vector{Vector{Float64}}  # formerly Xi
    c::Vector{(Matrix{Float64}, Array{Float64, 3})}  # formerly Omega, Lambda
end

# A vector of variational parameters.  The outer index is
# of celestial objects, and the inner index is over individual
# parameters for that object (referenced using ParamIndex).

typealias VariationalParams Vector{Vector{Float64}}
typealias RectVariationalParams Vector{Vector{Float64}}
typealias FreeVariationalParams Vector{Vector{Float64}}

#########################################################

immutable ParamIndex
    # A data structure to index parameters within
    # a VariationalParams object.

    # Variational parameter for a_s.
    # The probability of being a particular celestial object (Ia x 1 vector).
    # (Currently the probability of being a star or galaxy, respectively.)
    chi::Vector{Int64}

    # The location of the object (2x1 vector).
    mu::Vector{Int64}

    # Ia x 1 scalar variational parameters for r_s.  The first
    # row is for stars, and the second for galaxies
    gamma::Vector{Int64}
    zeta::Vector{Int64}

    # The weight given to a galaxy of type 1.
    theta::Int64

    # galaxy minor/major ratio
    rho::Int64

    # galaxy angle
    phi::Int64 

    # galaxy scale
    sigma::Int64

    # The remaining parameters are matrices where the
    # first column is for stars and the second is for galaxies.

    # DxI matrix of color prior component indicators.
    kappa::Array{Int64, 2}

    # (B - 1)xI matrices containing c_s means and variances, respectively.
    beta::Array{Int64, 2}
    lambda::Array{Int64, 2}

    # The size (largest index) of the parameterization.
    size::Int64
end

immutable UnconstrainedParamIndex
    # A data structure to index parameters within
    # an unconstrained VariationalParams object.

    # Unconstrained parameter for a_s of length (Ia - 1)
    # (the probability of being a type of celestial object).
    chi::Vector{Int64}

    # The location of the object (2x1 vector).
    mu::Vector{Int64}

    # Ix1 scalar variational parameters for r_s.  The first
    # row is for stars, and the second for galaxies
    gamma::Vector{Int64}
    zeta::Vector{Int64}

    # The weight given to a galaxy of type 1.
    theta::Int64

    # galaxy minor/major ratio
    rho::Int64

    # galaxy angle
    phi::Int64 

    # galaxy scale
    sigma::Int64

    # The remaining parameters are matrices where the
    # first column is for stars and the second is for galaxies.

    # Dx(Ia - 1) matrix of color prior component indicators.
    kappa::Array{Int64, 2}

    # (B - 1)xI matrices containing c_s means and variances, respectively.
    beta::Array{Int64, 2}
    lambda::Array{Int64, 2}

    # The size (largest index) of the parameterization.
    size::Int64
end

function get_param_ids()
    # Build a ParamIndex object.

    kappa_end = 12 + Ia * D
    beta_end = kappa_end + Ia * (B - 1)
    lambda_end = beta_end + Ia * (B - 1)

    kappa_ids = reshape([13 : kappa_end], D, Ia)
    beta_ids = reshape([kappa_end + 1 : beta_end], B - 1, Ia)
    lambda_ids = reshape([beta_end + 1 : lambda_end], B - 1, Ia)

    ParamIndex([1, 2], # chi
               [3, 4], # mu
               [5, 6], # gamma
               [7, 8], # zeta
               9, 10, 11, 12, # theta, rho, phi, sigma
               kappa_ids, beta_ids, lambda_ids,
               lambda_ids[end])
end

function get_unconstrained_param_ids()
    # Build a UnconstrainedParamIndex object.  Later the dimensions
    # of the unconstrained parameterizatoin may differ from the
    # constrained one (e.g. with simplicial constraints).
    #
    # Currently every alternative parameterization has the same
    # dimension in each parameter.

    # The last colunn of kappa is constrianed by the first Ia - 1 columns. 
    kappa_end = 11 + (Ia - 1) * D
    beta_end = kappa_end + Ia * (B - 1)
    lambda_end = beta_end + Ia * (B - 1)

    kappa_ids = reshape([12 : kappa_end], Ia - 1, D)
    beta_ids = reshape([kappa_end + 1 : beta_end], B - 1, Ia)
    lambda_ids = reshape([beta_end + 1 : lambda_end], B - 1, Ia)

    UnconstrainedParamIndex([1],    # chi
                            [2, 3], # mu
                            [4, 5], # gamma
                            [6, 7], # zeta
                            8, 9, 10, 11, # theta, rho, phi, sigma
                            kappa_ids, beta_ids, lambda_ids,
                            lambda_ids[end])
end

const ids = get_param_ids()
const ids_free = get_unconstrained_param_ids()

const all_params = [1:ids.size]
const all_params_free = [1:ids_free.size]

const star_pos_params = ids.mu
const galaxy_pos_params = [ids.mu, ids.theta, ids.rho, ids.phi, ids.sigma]

type ModelParams
    # The parameters for a particular image.
    #
    # Attributes:
    #  - vp: The variational parameters
    #  - vp_free: The unconstrained variational parameters.
    #  - pp: The prior parameters
    #  - patches: A vector of SkyPatch objects
    #  - tile_width: The number of pixels across a tile
    #  - S: The number of sources.

    # The following meanings are clear from the names.
    vp::VariationalParams
    pp::PriorParams
    patches::Vector{SkyPatch}
    tile_width::Int64

    # The number of sources.
    S::Int64

    ModelParams(vp, pp, patches, tile_width) = begin
        # There must be one patch for each celestial object.
        @assert length(vp) == length(patches)
        new(vp, pp, patches, tile_width, length(vp))
    end
end

#########################################################

type SensitiveFloat
    # A function value and its derivative with respect to its arguments.
    #
    # Attributes:
    #   v: The value
    #   d: The derivative with respect to each variable in
    #      P-dimensional VariationalParams for each of S celestial objects
    #      in a local_P x local_S matrix.
    #   h: The second derivative with respect to each variational parameter,
    #      in the same format as d.
    #   source_index: local_S x 1 vector of source ids with nonzero derivatives.
    #   param_index: local_P x 1 vector of parameter indices with
    #      nonzero derivatives. 
    #
    #  All derivatives not in source_index and param_index are zero.

    v::Float64
    d::Matrix{Float64} # local_P x local_S
    h::Matrix{Float64} # local_P x local_S
    source_index::Vector{Int64}
    param_index::Vector{Int64}
end

#########################################################

function zero_sensitive_float(s_index::Vector{Int64}, p_index::Vector{Int64})
    d = zeros(length(p_index), length(s_index))
    h = zeros(length(p_index), length(s_index))
    SensitiveFloat(0., d, h, s_index, p_index)
end

function clear!(sp::SensitiveFloat)
    sp.v = 0.
    fill!(sp.d, 0.)
end

#########################################################

end

