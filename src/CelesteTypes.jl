module CelesteTypes

VERSION < v"0.4.0-dev" && using Docile

export CatalogEntry
export band_letters

export Image, Blob, SkyPatch, ImageTile, PsfComponent, RawPSFComponents
export GalaxyComponent, GalaxyPrototype, galaxy_prototypes
export effective_radii

export ModelParams, PriorParams, UnconstrainedParams
export CanonicalParams, BrightnessParams, StarPosParams, GalaxyPosParams
export VariationalParams, FreeVariationalParams, RectVariationalParams

export shape_standard_alignment, brightness_standard_alignment, align

export SensitiveFloat

export zero_sensitive_float, clear!

export print_params
export ids, ids_free, star_ids, gal_ids
export ids_names, ids_free_names
export D, B, Ia

using Util

import Base.convert
import FITSIO
import Distributions
import WCSLIB
import ForwardDiff

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
All the information form a psField file needed to compute a raw PSF for a point.

Attributes:
 - rrows: A matrix of flattened eigenimages.
 - rnrow: The number of rows in an eigenimage.
 - rncol: The number of columns in an eigenimage.
 - cmat: The coefficients of the weight polynomial (see get_psf_at_point()).
""" ->
immutable RawPSFComponents
    rrows::Array{Float64,2}
    rnrow::Int32
    rncol::Int32
    cmat::Array{Float64,3}
end


@doc """
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

    # # Field-varying parameters.
    constant_background::Bool
    epsilon_mat::Array{Float64, 2}
    iota_vec::Array{Float64, 1}
    raw_psf_comp::RawPSFComponents
end

# Initialization for an image with noise and background parameters that are constant
# across the image.
Image(H::Int64, W::Int64, pixels::Matrix{Float64}, b::Int64, wcs::WCSLIB.wcsprm,
      epsilon::Float64, iota::Float64, psf::Vector{PsfComponent},
      run_num::Int64, camcol_num::Int64, field_num::Int64) = begin
    empty_psf_comp = RawPSFComponents(Array(Float64, 0, 0), -1, -1, Array(Float64, 0, 0, 0))
    Image(H, W, pixels, b, wcs, epsilon, iota, psf, run_num, camcol_num, field_num,
          true, Array(Float64, 0, 0), Array(Float64, 0), empty_psf_comp)
end

# Initialization for an image with noise and background parameters that vary across the image.
Image(H::Int64, W::Int64, pixels::Matrix{Float64}, b::Int64, wcs::WCSLIB.wcsprm,
      epsilon_mat::Array{Float64, 1}, iota_vec::Array{Float64, 2},
       psf::Vector{PsfComponent}, raw_psf_comp::RawPSFComponents,
      run_num::Int64, camcol_num::Int64, field_num::Int64) = begin
    Image(H, W, pixels, b, wcs, 0.0, 0.0, psf, run_num, camcol_num, field_num,
          false, epsilon_mat, iota_vec, raw_psf_comp)
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

@doc """
Attributes of the patch of sky surrounding a single
celestial object.

The amount of sky affected by a source in
world coordinates and an L_{\infty} norm.
""" ->
immutable SkyPatch
    center::Vector{Float64}
    radius::Float64
   
    psf::Vector{PsfComponent}
    wcs_jacobian::Array{Float64, 2}
end

SkyPatch(center::Vector{Float64}, radius::Float64) = begin
    SkyPatch(center, radius, PsfComponent[], Array(Float64, 0, 0))
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

typealias VariationalParams{NumType <: Number} Vector{Vector{NumType}}
typealias RectVariationalParams{NumType <: Number} Vector{Vector{NumType}}
typealias FreeVariationalParams{NumType <: Number} Vector{Vector{NumType}}

#########################################################

abstract ParamSet

# The variable names are:
# u       = Location in world coordinates (formerly mu)
# e_dev   = Weight given to a galaxy of type 1 (formerly theta)
# e_axis  = Galaxy minor/major ratio (formerly rho)
# e_angle = Galaxy angle (formerly phi)
# e_scale = Galaxy scale (sigma)
# For r1 and r2, the first row is stars, and the second is galaxies.
# r1      = Iax1 shape parameter for r_s. (formerly gamma)
# r2      = Iax1 scale parameter for r_s. (formerly zeta)
# c1      = C_s means (formerly beta)
# c2      = C_s variances (formerly lambda) 
# a       = probability of being a star or galaxy.  a[1] is the
#           probability of being a star and a[2] of being a galaxy. (formerly chi)
# k       = {D|D-1}xIa matrix of color prior component indicators. (formerly kappa)

# Parameters for location and galaxy shape.
ue_params = ((:u, 2), (:e_dev, 1), (:e_axis, 1), (:e_angle, 1),
        (:e_scale, 1))

# Parameters for the colors.
rc_params1 = ((:r1, 1), (:r2, 1), (:c1, B - 1), (:c2, B - 1))
rc_params2 = ((:r1, Ia), (:r2, Ia), (:c1, (B - 1,  Ia)),
        (:c2, (B - 1,  Ia)))

# Simplicial variables, either in constrained or free parameterizations.
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

# TODO: maybe these should be incorporated into the framework above (which I don't really understand.)
ids_free_names = Array(ASCIIString, length(ids_free))
for (name in names(ids_free)) 
    inds = ids_free.(name)
    for i = 1:length(inds)
        ids_free_names[inds[i]] = "$(name)_$(i)"
    end
end

ids_names = Array(ASCIIString, length(ids))
for (name in names(ids)) 
    inds = ids.(name)
    for i = 1:length(inds)
        ids_names[inds[i]] = "$(name)_$(i)"
    end
end


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
type ModelParams{NumType <: Number}
    vp::VariationalParams{NumType}
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

ModelParams{NumType <: Number}(vp::VariationalParams{NumType}, pp::PriorParams,
                               patches::Vector{SkyPatch}, tile_width::Int64) = begin
    ModelParams{NumType}(vp, pp, patches, tile_width)
end

function convert(::Type{ModelParams{ForwardDiff.Dual}}, mp::ModelParams{Float64})
    ModelParams(convert(Array{Array{ForwardDiff.Dual{Float64}, 1}, 1}, mp.vp),
                mp.pp, mp.patches, mp.tile_width)
end

@doc """
Display model parameters with the variable names.
""" ->
function print_params(mp::ModelParams)
    for s in 1:mp.S
        println("=======================\n Object $(s):")
        for var_name in names(ids)
            println(var_name)
            println(mp.vp[s][ids.(var_name)])
        end
    end
end

@doc """
Display several model parameters side by side.
""" ->
function print_params(mp_tuple::ModelParams...)
    println("Printing for $(length(mp_tuple)) parameters.")
    for s in 1:mp_tuple[1].S
        println("=======================\n Object $(s):")
        for var_name in names(ids)
            println(var_name)
            mp_vars = [ collect(mp_tuple[index].vp[s][ids.(var_name)]) for index in 1:length(mp_tuple) ] 
            println(reduce(hcat, mp_vars))
        end
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
type SensitiveFloat{ParamType <: ParamSet, NumType <: Number}
    v::NumType
    d::Matrix{NumType} # local_P x local_S
    h::Matrix{NumType} # local_P x local_S
    ids::ParamType
end

#########################################################

function zero_sensitive_float{ParamType <: ParamSet}(::Type{ParamType}, NumType::DataType, local_S::Int64)
    local_P = length(ParamType)
    d = zeros(NumType, local_P, local_S)
    h = zeros(NumType, local_P, local_S)
    SensitiveFloat{ParamType, NumType}(zero(NumType), d, h, getids(ParamType))
end

function zero_sensitive_float{ParamType <: ParamSet}(::Type{ParamType}, NumType::DataType)
    # Default to a single source.
    zero_sensitive_float(ParamType, NumType, 1)
end

function clear!{ParamType <: ParamSet, NumType <: Number}(sp::SensitiveFloat{ParamType, NumType})
    sp.v = zero(NumType)
    fill!(sp.d, zero(NumType))
end

# If no type is specified, default to using Float64.
function zero_sensitive_float{ParamType <: ParamSet}(param_arg::Type{ParamType}, local_S::Int64)
    zero_sensitive_float(param_arg, Float64, local_S)
end

function zero_sensitive_float{ParamType <: ParamSet}(param_arg::Type{ParamType})
    zero_sensitive_float(param_arg, Float64, 1)
end


#########################################################

end

