module CelesteTypes

VERSION < v"0.4.0-dev" && using Docile

export CatalogEntry
export band_letters

export Image, Blob, SkyPatch, ImageTile, PsfComponent
export GalaxyComponent, GalaxyPrototype, galaxy_prototypes
export effective_radii

export ModelParams, PriorParams, UnconstrainedParams
export CanonicalParams, BrightnessParams, StarPosParams, GalaxyPosParams
export VariationalParams, FreeVariationalParams, RectVariationalParams

export shape_standard_alignment, brightness_standard_alignment, align

export SensitiveFloat

export zero_sensitive_float, clear!

export ids, ids_free, star_ids, gal_ids
export D, B, Ia

using Util

import FITSIO
import Distributions
import WCSLIB

import Base.length

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

@doc """
Parameters of a single normal component of a galaxy.

Attributes:
  etaBar: The weight of the galaxy component
  nuBar: The scale of the galaxy component
""" ->
immutable GalaxyComponent
    etaBar::Float64
    nuBar::Float64
end

typealias GalaxyPrototype Vector{GalaxyComponent}

@doc """
Pre-defined shapes for galaxies.

Returns:
  dev_prototype: An array of GalaxyComponent for de Vaucouleurs galaxy types
  exp_prototype: An array of GalaxyComponent for exponenttial galaxy types
""" ->
function get_galaxy_prototypes()
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


@doc """
A single normal component of the point spread function.

Args:
  alphaBar: The scalar weight of the component.
  xiBar: The 2x1 location vector
  tauBar: The 2x2 covariance

Attributes:
  alphaBar: The scalar weight of the component.
  xiBar: The 2x1 location vector
  tauBar: The 2x2 covariance (tau_bar in the ICLM paper)
  tauBarInv: The 2x2 precision
  tauBarLd: The log determinant of the covariance
""" ->
immutable PsfComponent
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

@doc """An image, taken though a particular filter band""" ->
type Image
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

@doc """
Tiles of pixels that share the same set of
relevant sources (or other calculations).

These are in tile coordinates --- not pixel or sky coordinates.
(I.e., the range of hh and ww are the number of horizontal
 and vertical tiles in the image, respectively.)
""" ->
immutable ImageTile
    hh::Int64
    ww::Int64
    img::Image
end

@doc """A vector of images, one for each filter band""" ->
typealias Blob Vector{Image}

@doc """The amount of sky affected by a source""" ->
immutable SkyPatch #pixel coordinates for now, soon wcs
    center::Vector{Float64}
    radius::Float64
end


#########################################################

immutable PriorParams
    a::Vector{Float64}  # formerly Phi
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

abstract ParamSet


ue_params = ((:u, 2), (:e_dev, 1), (:e_axis, 1), (:e_angle, 1),
        (:e_scale, 1))
rc_params1 = ((:r1, 1), (:r2, 1), (:c1, B - 1), (:c2, B - 1))
rc_params2 = ((:r1, Ia), (:r2, Ia), (:c1, (B - 1,  Ia)),
        (:c2, (B - 1,  Ia)))
ak_simplex = ((:a, Ia), (:k, (D, Ia)))
ak_free = ((:a, Ia - 1), (:k, (D - 1, Ia)))

const param_specs = [
    (:StarPosParams, :star_ids, ((:u, 2),)),
    (:GalaxyPosParams, :gal_ids, ue_params),
    (:BrightnessParams, :bids, rc_params1),
    (:CanonicalParams, :ids, tuple(ue_params..., rc_params2..., ak_simplex...)),
    (:UnconstrainedParams, :ids_free, tuple(ue_params..., rc_params2..., ak_free...)),
]

for (pn, ids_name, pf) in param_specs
    ids_fields = Any[]
    ids_init = Any[]

    prev_end = 0
    for (n, ll) in pf
        id_field = ll == 1 ? :(Int64) : :(Array{Int64, $(length(ll))})
        push!(ids_fields, :($n::$id_field))

        field_len = *(ll...)

        ids_array = ll == 1 ? prev_end + 1 :
            [(prev_end + 1) : (prev_end + field_len)]
        if length(ll) >= 2
            ids_array = :(reshape($ids_array, $ll))
        end
        push!(ids_init, ids_array)

        prev_end += field_len
    end

    new_call = Expr(:call, :new, ids_init...)
    constructor = Expr(:(=),:($pn()), new_call)
    push!(ids_fields, constructor)
    fields_block = Expr(:block, ids_fields...)
    struct_sig = Expr(:(<:), pn, :ParamSet)
    struct_dec = Expr(:type, false, struct_sig, fields_block)
    eval(struct_dec)

    eval(:(const $ids_name = $pn()))
    eval(:(getids(::Type{$pn}) = $ids_name))
    eval(:(length(::Type{$pn}) = $prev_end))
    eval(:(length(an_ids::$pn) = $prev_end))
end


#TODO: build these from ue_align, etc., here.
align(::StarPosParams, CanonicalParams) = ids.u
align(::GalaxyPosParams, CanonicalParams) = 
   [ids.u; ids.e_dev; ids.e_axis; ids.e_angle; ids.e_scale]
align(::CanonicalParams, CanonicalParams) = [1:length(CanonicalParams)]

const shape_standard_alignment = (ids.u,
   [ids.u; ids.e_dev; ids.e_axis; ids.e_angle; ids.e_scale])
bright_ids(i) = [ids.r1[i]; ids.r2[i]; ids.c1[:, i]; ids.c2[:, i]]
const brightness_standard_alignment = (bright_ids(1), bright_ids(2))

#########################################################

@doc """
The parameters for a particular image.

Attributes:
 - vp: The variational parameters
 - pp: The prior parameters
 - patches: A vector of SkyPatch objects
 - tile_width: The number of pixels across a tile
 - S: The number of sources.
""" ->
type ModelParams
    vp::VariationalParams
    pp::PriorParams
    patches::Vector{SkyPatch}
    tile_width::Int64

    S::Int64

    ModelParams(vp, pp, patches, tile_width) = begin
        # There must be one patch for each celestial object.
        @assert length(vp) == length(patches)
        new(vp, pp, patches, tile_width, length(vp))
    end
end

#########################################################

@doc """
A function value and its derivative with respect to its arguments.

Attributes:
  v: The value
  d: The derivative with respect to each variable in
     P-dimensional VariationalParams for each of S celestial objects
     in a local_P x local_S matrix.
  h: The second derivative with respect to each variational parameter,
     in the same format as d.
""" ->
type SensitiveFloat{T <: ParamSet}
    v::Float64
    d::Matrix{Float64} # local_P x local_S
    h::Matrix{Float64} # local_P x local_S
    ids::T
end

#########################################################

function zero_sensitive_float{T <: ParamSet}(::Type{T})
    zero_sensitive_float(T, 1)
end

function zero_sensitive_float{T <: ParamSet}(::Type{T}, local_S::Int64)
    local_P = length(T)
    d = zeros(local_P, local_S)
    h = zeros(local_P, local_S)
    SensitiveFloat{T}(0., d, h, getids(T))
end

function clear!(sp::SensitiveFloat)
    sp.v = 0.
    fill!(sp.d, 0.)
end

#########################################################

end

