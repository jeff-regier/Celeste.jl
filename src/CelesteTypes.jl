module CelesteTypes

VERSION < v"0.4.0-dev" && using Docile

export CatalogEntry
export band_letters

export Image, Blob, TiledImage, TiledBlob, ImageTile, SkyPatch, PsfComponent
export GalaxyComponent, GalaxyPrototype, galaxy_prototypes
export effective_radii

export ModelParams, PriorParams, UnconstrainedParams
export CanonicalParams, BrightnessParams, StarPosParams, GalaxyPosParams
export VariationalParams, FreeVariationalParams, RectVariationalParams

export shape_standard_alignment, brightness_standard_alignment, align

export SensitiveFloat, zero_sensitive_float, clear!
export print_params

export ids, ids_free, star_ids, gal_ids
export ids_names, ids_free_names
export D, B, Ia

export set_hess!

export print_params

using Util
using SloanDigitalSkySurvey.PSF.RawPSFComponents
using Compat
using ForwardDiff

import Base.convert
import Base.+
import Distributions
import FITSIO
import DualNumbers
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
    objid::ASCIIString
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
        4.26347652e-2, 2.40127183e-1, 6.85907632e-1, 1.51937350,
        2.83627243, 4.46467501, 5.72440830, 5.60989349]
    dev_amp /= sum(dev_amp)
    dev_var = [
        2.23759216e-4, 1.00220099e-3, 4.18731126e-3, 1.69432589e-2,
        6.84850479e-2, 2.87207080e-1, 1.33320254, 8.40215071]

	exp_amp = [
        2.34853813e-3, 3.07995260e-2, 2.23364214e-1,
        1.17949102, 4.33873750, 5.99820770]
    exp_amp /= sum(exp_amp)
    exp_var = [
        1.20078965e-3, 8.84526493e-3, 3.91463084e-2,
        1.39976817e-1, 4.60962500e-1, 1.50159566]

	# Adjustments to the effective radius hard-coded above.
	# (The effective radius is the distance from the center containing half
  # the light.)
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

# Initialization for an image with noise and background parameters that are
# constant across the image.
Image(H::Int64, W::Int64, pixels::Matrix{Float64}, b::Int64, wcs::WCSLIB.wcsprm,
      epsilon::Float64, iota::Float64, psf::Vector{PsfComponent},
      run_num::Int64, camcol_num::Int64, field_num::Int64) = begin
    empty_psf_comp =
      RawPSFComponents(Array(Float64, 0, 0), -1, -1, Array(Float64, 0, 0, 0))
    Image(H, W, pixels, b, wcs, epsilon, iota, psf,
          run_num, camcol_num, field_num,
          true, Array(Float64, 0, 0), Array(Float64, 0), empty_psf_comp)
end

# Initialization for an image with noise and background parameters that vary
# across the image.
Image(H::Int64, W::Int64, pixels::Matrix{Float64}, b::Int64, wcs::WCSLIB.wcsprm,
      epsilon_mat::Array{Float64, 1}, iota_vec::Array{Float64, 2},
       psf::Vector{PsfComponent}, raw_psf_comp::RawPSFComponents,
      run_num::Int64, camcol_num::Int64, field_num::Int64) = begin
    Image(H, W, pixels, b, wcs, 0.0, 0.0, psf, run_num, camcol_num, field_num,
          false, epsilon_mat, iota_vec, raw_psf_comp)
end


@doc """A vector of images, one for each filter band""" ->
typealias Blob Vector{Image}


@doc """
Tiles of pixels that share the same set of
relevant sources (or other calculations).  It contains all the information
necessary to compute the ELBO and derivatives in this patch of sky.

Note that this cannot be of type Image for the purposes of Celeste
because the raw wcs object is a C++ pointer which julia resonably
refuses to parallelize.

Attributes:
- hh: The tile row index (in 1:number of tile rows)
- ww: The tile column index (in 1:number of tile columns)
- h_range: The h pixel locations of the tile in the original image
- w_range: The w pixel locations of the tile in the original image
- h_width: The width of the tile in the h direction
- w_width: The width of the tile in the w direction
- pixels: The pixel values
- remainder: the same as in the Image type.
""" ->
immutable ImageTile
    hh::Int64
    ww::Int64
    b::Int64

    h_range::UnitRange{Int64}
    w_range::UnitRange{Int64}
    h_width::Int64
    w_width::Int64
    pixels::Matrix{Float64}

    constant_background::Bool
    epsilon::Float64
    epsilon_mat::Matrix{Float64}
    iota::Float64
    iota_vec::Vector{Float64}
end


@doc """
Return the range of image pixels in an ImageTile.

Args:
  - hh: The tile row index (in 1:number of tile rows)
  - ww: The tile column index (in 1:number of tile columns)
  - H: The number of pixel rows in the image
  - W: The number of pixel columns in the image
  - tile_width: The width and height of a tile in pixels
""" ->
function tile_range(hh::Int64, ww::Int64, H::Int64, W::Int64, tile_width::Int64)
    h1 = 1 + (hh - 1) * tile_width
    h2 = min(hh * tile_width, H)
    w1 = 1 + (ww - 1) * tile_width
    w2 = min(ww * tile_width, W)
    h1:h2, w1:w2
end


@doc """
Constructs an image tile from an image.

Args:
  - img: The Image to be broken into tiles
  - hh: The tile row index (in 1:number of tile rows)
  - ww: The tile column index (in 1:number of tile columns)
  - tile_width: The width and height of a tile in pixels
""" ->
ImageTile(hh::Int64, ww::Int64, img::Image, tile_width::Int64) = begin
  h_range, w_range = tile_range(hh, ww, img.H, img.W, tile_width)
  ImageTile(img, h_range, w_range; hh=hh, ww=ww)
end

@doc """
Constructs an image tile from specific image pixels.

Args:
  - img: The Image to be broken into tiles
  - h_range: A UnitRange for the h pixels
  - w_range: A UnitRange for the w pixels
  - hh: Optional h index in tile coordinates
  - ww: Optional w index in tile coordinates
""" ->
ImageTile(img::Image,
          h_range::UnitRange{Int64}, w_range::UnitRange{Int64};
          hh::Int64=1, ww::Int64=1) = begin
  b = img.b
  h_width = maximum(h_range) - minimum(h_range) + 1
  w_width = maximum(w_range) - minimum(w_range) + 1
  pixels = img.pixels[h_range, w_range]

  if img.constant_background
    epsilon_mat = img.epsilon_mat
    iota_vec = img.iota_vec
  else
    # TODO: this subsetting doesn't seem to be working.
    epsilon_mat = img.epsilon_mat[h_range, w_range]
    iota_vec = img.iota_vec[h_range]
  end

  ImageTile(hh, ww, b, h_range, w_range, h_width, w_width, pixels,
            img.constant_background, img.epsilon, epsilon_mat,
            img.iota, iota_vec)
end


typealias TiledImage Array{ImageTile, 2}
typealias TiledBlob Vector{TiledImage}


@doc """
Attributes of the patch of sky surrounding a single
celestial object in a single image.

Attributes:
  - center: The approximate source location in world coordinates
  - radius: The width of the influence of the object in world coordinates

  - psf: The point spread function in this region of the sky
  - wcs_jacobian: The jacobian of the WCS transform in this region of the
                  sky for each band
  - pixel_center: The pixel location of center in each band.
""" ->
immutable SkyPatch
    center::Vector{Float64}
    radius::Float64

    psf::Vector{PsfComponent}
    wcs_jacobian::Matrix{Float64}
    pixel_center::Vector{Float64}
end


#########################################################

immutable PriorParams
    a::Vector{Float64}  # formerly Phi
    r_mean::Vector{Float64}
    r_var::Vector{Float64}
    k::Matrix{Float64}  # formerly Xi
    c_mean::Array{Float64, 3} # formerly Omega
    c_cov::Array{Float64, 4} # formerly Lambda
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
#           probability of being a star and a[2] of being a galaxy.
#           (formerly chi)
# k       = {D|D-1}xIa matrix of color prior component indicators.
#           (formerly kappa)

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
    (:UnconstrainedParams, :ids_free,
     tuple(ue_params..., rc_params2..., ak_free...)),
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
            collect( (prev_end + 1):(prev_end + field_len) )
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
align(::CanonicalParams, CanonicalParams) = collect(1:length(CanonicalParams))

# The shape and brightness parameters for stars and galaxies respectively.
const shape_standard_alignment = (ids.u,
   [ids.u; ids.e_dev; ids.e_axis; ids.e_angle; ids.e_scale])
bright_ids(i) = [ids.r1[i]; ids.r2[i]; ids.c1[:, i]; ids.c2[:, i]]
const brightness_standard_alignment = (bright_ids(1), bright_ids(2))

# TODO: maybe these should be incorporated into the framework above
# (which I don't really understand.)
function get_id_names(
  ids::@compat(Union{CanonicalParams, UnconstrainedParams}))
  ids_names = Array(ASCIIString, length(ids))
  for (name in fieldnames(ids))
    inds = ids.(name)
    if length(size(inds)) == 0
      ids_names[inds] = "$(name)"
    elseif length(size(inds)) == 1
      for i = 1:size(inds)[1]
          ids_names[inds[i]] = "$(name)_$(i)"
      end
    elseif length(size(inds)) == 2
      for i = 1:size(inds)[1], j = 1:size(inds)[2]
          ids_names[inds[i, j]] = "$(name)_$(i)_$(j)"
      end
    else
      error("Names of 3d parameters not supported ($(name))")
    end
  end
  return ids_names
end

const ids_names = get_id_names(ids)
const ids_free_names = get_id_names(ids_free)

#########################################################

@doc """
The parameters for a particular image.

Attributes:
 - vp: The variational parameters
 - pp: The prior parameters
 - patches: An (objects X bands) matrix of SkyPatch objects
 - tile_width: The number of pixels across a tile
 - tile_sources: A vector (over bands) of an array (over tiles) of vectors
                 of sources influencing each tile.
 - active_sources: Indices of the sources that are currently being fit by the
                   model.
 - objids: Global object ids for the sources in this ModelParams object.

 - S: The number of sources.
""" ->
type ModelParams{NumType <: Number}
    vp::VariationalParams{NumType}
    pp::PriorParams
    patches::Array{SkyPatch, 2}
    tile_sources::Vector{Array{Array{Int64}}}
    active_sources::Vector{Int64}
    objids::Vector{ASCIIString}

    S::Int64

    ModelParams(vp, pp) = begin
        # There must be one patch for each celestial object.
        S = length(vp)
        all_tile_sources = fill(fill(collect(1:S), 1, 1), 5)
        patches = Array(SkyPatch, S, 5)
        active_sources = collect(1:S)
        objids = ASCIIString[string(s) for s in 1:S]

        new(vp, pp, patches, all_tile_sources, active_sources, objids, S)
    end
end

# TODO: Is this second initialization function necessary?
ModelParams{NumType <: Number}(
  vp::VariationalParams{NumType}, pp::PriorParams) = begin
    ModelParams{NumType}(vp, pp)
end

function convert(::Type{ModelParams{DualNumbers.Dual{Float64}}},
                 mp::ModelParams{Float64})
    mp_dual =
      ModelParams(convert(Array{Array{DualNumbers.Dual{Float64}, 1}, 1}, mp.vp),
                  mp.pp)
    mp_dual.patches = mp.patches
    mp_dual.tile_sources = mp.tile_sources
    mp_dual.active_sources = mp.active_sources
    mp_dual.objids = mp.objids
    mp_dual
end


# Does this make any sense?
function convert(::Type{ForwardDiff.GradientNumber},
                 mp::ModelParams{Float64})

    x = mp.vp[1]
    P = length(x)
    FDType = ForwardDiff.GradientNumber{length(mp.vp[1]), Float64}
    fd_x = [ ForwardDiff.GradientNumber(x[i], zeros(Float64, P)...) for i=1:P ]
    convert(FDType, x[1])

    vp_fd = convert(Array{Array{FDType, 1}, 1}, mp.vp[1])
    mp_fd = ModelParams(vp_fd, mp.pp)
    mp_fd.patches = mp.patches
    mp_fd.tile_sources = mp.tile_sources
    mp_fd.active_sources = mp.active_sources
    mp_fd.objids = mp.objids
    mp_fd
end


@doc """
Display model parameters with the variable names.
""" ->
function print_params(mp::ModelParams)
    for s in mp.active_sources
        println("=======================\n Object $(s):")
        for var_name in fieldnames(ids)
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
    for s in mp_tuple[1].active_sources
        println("=======================\n Object $(s):")
        for var_name in fieldnames(ids)
            println(var_name)
            mp_vars =
              [ collect(mp_tuple[index].vp[s][ids.(var_name)]) for
                index in 1:length(mp_tuple) ]
            println(reduce(hcat, mp_vars))
        end
    end
end


@doc """
Display a Celeste catalog entry.
""" ->
function print_cat_entry(cat_entry::CatalogEntry)
    [println("$name: $(cat_entry.(name))") for name in
            fieldnames(cat_entry)]
end


type HessianEntry{NumType <: Number}
  # An entry in a sparse Hessian containing a derivative with respect
  # to source s1, parameter i1 and source s2, parameter i2,
  # with value v.
  s1::Int64
  i1::Int64
  s2::Int64
  i2::Int64
  v::NumType
end


@doc """
A function value and its derivative with respect to its arguments.

Attributes:
  v:  The value
  d:  The derivative with respect to each variable in
      P-dimensional VariationalParams for each of S celestial objects
      in a local_P x local_S matrix.
  h:  The second derivative with respect to each variational parameter,
      in the same format as d.  This is used for the full Hessian
      with respect to all the sources.
  hs: An array of per-source Hessians.  This will generally be reserved
      for the Hessian of brightness values that depend only on one source.
""" ->
type SensitiveFloat{ParamType <: CelesteTypes.ParamSet, NumType <: Number}
    v::NumType
    d::Matrix{NumType} # local_P x local_S
    h::Array{HessianEntry{NumType}}
    hs::Array{Matrix{NumType}} #  local_S array of local_P x local_P
    ids::ParamType
end


# TODO: wrap this into its own module
include(joinpath(Pkg.dir("Celeste"), "src/SensitiveFloat.jl"))

end
