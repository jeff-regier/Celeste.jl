module CelesteTypes

export CatalogEntry
export band_letters

export Image, Blob, TiledImage, TiledBlob, ImageTile, SkyPatch, PsfComponent
export GalaxyComponent, GalaxyPrototype, galaxy_prototypes
export effective_radii

export ModelParams, PriorParams, UnconstrainedParams
export CanonicalParams, BrightnessParams, StarPosParams
export GalaxyPosParams, GalaxyShapeParams
export VariationalParams, FreeVariationalParams, RectVariationalParams

export shape_standard_alignment, brightness_standard_alignment
export gal_shape_alignment, align

export SensitiveFloat, zero_sensitive_float, clear!
export print_params

export ids, ids_free, star_ids, gal_ids, gal_shape_ids, bids
export ids_names, ids_free_names
export D, B, Ia

export set_hess!, multiply_sfs!, combine_sfs!

export print_params

using Util
using SloanDigitalSkySurvey.PSF.RawPSFComponents

import Base.convert
import Base.+
import Distributions
import FITSIO
import WCS.WCSTransform
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
    objid::ASCIIString
end

############################################

"""
Parameters of a single normal component of a galaxy.

Attributes:
  etaBar: The weight of the galaxy component
  nuBar: The scale of the galaxy component
"""
immutable GalaxyComponent
    etaBar::Float64
    nuBar::Float64
end

typealias GalaxyPrototype Vector{GalaxyComponent}

"""
Pre-defined shapes for galaxies.

Returns:
  dev_prototype: An array of GalaxyComponent for de Vaucouleurs galaxy types
  exp_prototype: An array of GalaxyComponent for exponenttial galaxy types
"""
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

"""An image, taken though a particular filter band"""
type Image
    # The image height.
    H::Int

    # The image width.
    W::Int

    # An HxW matrix of pixel intensities.
    pixels::Matrix{Float64}

    # The band id (takes on values from 1 to 5).
    b::Int

    # World coordinates
    wcs::WCSTransform

    # The background noise in nanomaggies.
    epsilon::Float64

    # The expected number of photons contributed to this image
    # by a source 1 nanomaggie in brightness.
    iota::Float64

    # The components of the point spread function.
    psf::Vector{PsfComponent}

    # SDSS-specific identifiers. A field is a particular region of the sky.
    # A Camcol is the output of one camera column as part of a Run.
    run_num::Int
    camcol_num::Int
    field_num::Int

    # # Field-varying parameters.
    constant_background::Bool
    epsilon_mat::Array{Float64, 2}
    iota_vec::Array{Float64, 1}
    raw_psf_comp::RawPSFComponents
end

# Initialization for an image with noise and background parameters that are
# constant across the image.
function Image(H::Int, W::Int, pixels::Matrix{Float64}, b::Int,
               wcs::WCSTransform, epsilon::Float64, iota::Float64,
               psf::Vector{PsfComponent}, run_num::Int, camcol_num::Int,
               field_num::Int)
    empty_psf_comp = RawPSFComponents(Array(Float64, 0, 0), -1, -1,
                                      Array(Float64, 0, 0, 0))
    Image(H, W, pixels, b, wcs, epsilon, iota, psf,
          run_num, camcol_num, field_num,
          true, Array(Float64, 0, 0), Array(Float64, 0), empty_psf_comp)
end

# Initialization for an image with noise and background parameters that vary
# across the image.
function Image(H::Int, W::Int, pixels::Matrix{Float64}, b::Int,
               wcs::WCSTransform, epsilon_mat::Vector{Float64},
               iota_vec::Matrix{Float64}, psf::Vector{PsfComponent},
               raw_psf_comp::RawPSFComponents, run_num::Int, camcol_num::Int,
               field_num::Int)
    Image(H, W, pixels, b, wcs, 0.0, 0.0, psf, run_num, camcol_num,
          field_num, false, epsilon_mat, iota_vec, raw_psf_comp)
end


"""A vector of images, one for each filter band"""
typealias Blob Vector{Image}


"""
Tiles of pixels that share the same set of
relevant sources (or other calculations).  It contains all the information
necessary to compute the ELBO and derivatives in this patch of sky.

Note that this cannot be of type Image for the purposes of Celeste
because the raw wcs object is a C++ pointer which julia resonably
refuses to parallelize.

Attributes:
- h_range: The h pixel locations of the tile in the original image
- w_range: The w pixel locations of the tile in the original image
- h_width: The width of the tile in the h direction
- w_width: The width of the tile in the w direction
- pixels: The pixel values
- remainder: the same as in the Image type.
"""
immutable ImageTile
    b::Int

    h_range::UnitRange{Int}
    w_range::UnitRange{Int}
    h_width::Int
    w_width::Int
    pixels::Matrix{Float64}

    constant_background::Bool
    epsilon::Float64
    epsilon_mat::Matrix{Float64}
    iota::Float64
    iota_vec::Vector{Float64}
end


"""
Return the range of image pixels in an ImageTile.

Args:
  - hh: The tile row index (in 1:number of tile rows)
  - ww: The tile column index (in 1:number of tile columns)
  - H: The number of pixel rows in the image
  - W: The number of pixel columns in the image
  - tile_width: The width and height of a tile in pixels
"""
function tile_range(hh::Int, ww::Int, H::Int, W::Int, tile_width::Int)
    h1 = 1 + (hh - 1) * tile_width
    h2 = min(hh * tile_width, H)
    w1 = 1 + (ww - 1) * tile_width
    w2 = min(ww * tile_width, W)
    h1:h2, w1:w2
end


"""
Constructs an image tile from an image.

Args:
  - img: The Image to be broken into tiles
  - hh: The tile row index (in 1:number of tile rows)
  - ww: The tile column index (in 1:number of tile columns)
  - tile_width: The width and height of a tile in pixels
"""
function ImageTile(hh::Int, ww::Int, img::Image, tile_width::Int)
  h_range, w_range = tile_range(hh, ww, img.H, img.W, tile_width)
  ImageTile(img, h_range, w_range; hh=hh, ww=ww)
end

"""
Constructs an image tile from specific image pixels.

Args:
  - img: The Image to be broken into tiles
  - h_range: A UnitRange for the h pixels
  - w_range: A UnitRange for the w pixels
  - hh: Optional h index in tile coordinates
  - ww: Optional w index in tile coordinates
"""
function ImageTile(img::Image, h_range::UnitRange{Int},
                   w_range::UnitRange{Int}; hh::Int=1, ww::Int=1)
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

  ImageTile(b, h_range, w_range, h_width, w_width, pixels,
            img.constant_background, img.epsilon, epsilon_mat,
            img.iota, iota_vec)
end


typealias TiledImage Array{ImageTile, 2}
typealias TiledBlob Vector{TiledImage}


"""
Attributes of the patch of sky surrounding a single
celestial object in a single image.

Attributes:
  - center: The approximate source location in world coordinates
  - radius: The width of the influence of the object in world coordinates

  - psf: The point spread function in this region of the sky
  - wcs_jacobian: The jacobian of the WCS transform in this region of the
                  sky for each band
  - pixel_center: The pixel location of center in each band.
"""
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
# ParamSet types
#
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

abstract ParamSet

type StarPosParams <: ParamSet
    u::Vector{Int}
    StarPosParams() = new([1, 2])
end
const star_ids = StarPosParams()
getids(::Type{StarPosParams}) = star_ids
length(::Type{StarPosParams}) = 2


type GalaxyShapeParams <: ParamSet
    e_axis::Int
    e_angle::Int
    e_scale::Int
    GalaxyShapeParams() = new(1, 2, 3)
end
const gal_shape_ids = GalaxyShapeParams()
getids(::Type{GalaxyShapeParams}) = gal_shape_ids
length(::Type{GalaxyShapeParams}) = 3


type GalaxyPosParams <: ParamSet
    u::Vector{Int}
    e_dev::Int
    e_axis::Int
    e_angle::Int
    e_scale::Int
    GalaxyPosParams() = new([1, 2], 3, 4, 5, 6)
end
const gal_ids = GalaxyPosParams()
getids(::Type{GalaxyPosParams}) = gal_ids
length(::Type{GalaxyPosParams}) = 6


type BrightnessParams <: ParamSet
    r1::Int
    r2::Int
    c1::Vector{Int}
    c2::Vector{Int}
    BrightnessParams() = new(1, 2,
                             collect(3:(3+(B-1)-1)),
                             collect((3+B-1):(3+2*(B-1)-1)))
end
const bids = BrightnessParams()
getids(::Type{BrightnessParams}) = bids
length(::Type{BrightnessParams}) = 2 + 2 * (B-1)


type CanonicalParams <: ParamSet
    u::Vector{Int}
    e_dev::Int
    e_axis::Int
    e_angle::Int
    e_scale::Int
    r1::Vector{Int}
    r2::Vector{Int}
    c1::Matrix{Int}
    c2::Matrix{Int}
    a::Vector{Int}
    k::Matrix{Int}
    CanonicalParams() = 
        new([1, 2], 3, 4, 5, 6,
            collect(7:(7+Ia-1)),  # r1
            collect((7+Ia):(7+2Ia-1)), # r2
            reshape((7+2Ia):(7+2Ia+(B-1)*Ia-1), (B-1, Ia)),  # c1
            reshape((7+2Ia+(B-1)*Ia):(7+2Ia+2*(B-1)*Ia-1), (B-1, Ia)),  # c2
            collect((7+2Ia+2*(B-1)*Ia):(7+3Ia+2*(B-1)*Ia-1)),  # a
            reshape((7+3Ia+2*(B-1)*Ia):(7+3Ia+2*(B-1)*Ia+D*Ia-1), (D, Ia))) # k
end
const ids = CanonicalParams()
getids(::Type{CanonicalParams}) = ids
length(::Type{CanonicalParams}) = 6 + 3*Ia + 2*(B-1)*Ia + D*Ia


type UnconstrainedParams <: ParamSet
    u::Vector{Int}
    e_dev::Int
    e_axis::Int
    e_angle::Int
    e_scale::Int
    r1::Vector{Int}
    r2::Vector{Int}
    c1::Matrix{Int}
    c2::Matrix{Int}
    a::Vector{Int}
    k::Matrix{Int}
    UnconstrainedParams() = 
        new([1, 2], 3, 4, 5, 6,
            collect(7:(7+Ia-1)),  # r1
            collect((7+Ia):(7+2Ia-1)), # r2
            reshape((7+2Ia):(7+2Ia+(B-1)*Ia-1), (B-1, Ia)),  # c1
            reshape((7+2Ia+(B-1)*Ia):(7+2Ia+2*(B-1)*Ia-1), (B-1, Ia)),  # c2
            collect((7+2Ia+2*(B-1)*Ia):(7+2Ia+2*(B-1)*Ia+(Ia-1)-1)),  # a
            reshape((7+2Ia+2*(B-1)*Ia+(Ia-1)):
                    (7+2Ia+2*(B-1)*Ia+(Ia-1)+(D-1)*Ia-1), (D-1, Ia))) # k
end
const ids_free = UnconstrainedParams()
getids(::Type{UnconstrainedParams}) = ids_free
length(::Type{UnconstrainedParams}) =  6 + 2*Ia + 2*(B-1)*Ia + (D-1)*Ia + Ia-1


# define length(value) in addition to length(Type) for ParamSets
length{T<:ParamSet}(::T) = length(T)

#TODO: build these from ue_align, etc., here.
align(::StarPosParams, CanonicalParams) = ids.u
align(::GalaxyPosParams, CanonicalParams) =
   [ids.u; ids.e_dev; ids.e_axis; ids.e_angle; ids.e_scale]
align(::CanonicalParams, CanonicalParams) = collect(1:length(CanonicalParams))
align(::GalaxyShapeParams, GalaxyPosParams) =
  [gal_ids.e_axis; gal_ids.e_angle; gal_ids.e_scale]

# The shape and brightness parameters for stars and galaxies respectively.
const shape_standard_alignment = (ids.u,
   [ids.u; ids.e_dev; ids.e_axis; ids.e_angle; ids.e_scale])
bright_ids(i) = [ids.r1[i]; ids.r2[i]; ids.c1[:, i]; ids.c2[:, i]]
const brightness_standard_alignment = (bright_ids(1), bright_ids(2))

# Note that gal_shape_alignment aligns the shape ids with the GalaxyPosParams,
# not the CanonicalParams.
const gal_shape_alignment = align(gal_shape_ids, gal_ids)

# TODO: maybe these should be incorporated into the framework above
# (which I don't really understand.)
function get_id_names(
  ids::Union{CanonicalParams, UnconstrainedParams})
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

"""
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
"""
type ModelParams{T <: Number}
    vp::VariationalParams{T}
    pp::PriorParams
    patches::Array{SkyPatch, 2}
    tile_sources::Vector{Array{Array{Int}}}
    active_sources::Vector{Int}
    objids::Vector{ASCIIString}

    S::Int

    function ModelParams(vp::VariationalParams{T}, pp::PriorParams)
        # There must be one patch for each celestial object.
        S = length(vp)
        all_tile_sources = fill(fill(collect(1:S), 1, 1), 5)
        patches = Array(SkyPatch, S, 5)
        active_sources = collect(1:S)
        objids = ASCIIString[string(s) for s in 1:S]

        new(vp, pp, patches, all_tile_sources, active_sources, objids, S)
    end
end
ModelParams{T <: Number}(vp::VariationalParams{T}, pp::PriorParams) =
    ModelParams{T}(vp, pp)


function convert(FDType::Type{ForwardDiff.GradientNumber},
                 mp::ModelParams{Float64})
    x = mp.vp[1]
    P = length(x)
    FDType = ForwardDiff.GradientNumber{length(mp.vp[1]), Float64}

    fd_x = [ ForwardDiff.GradientNumber(x[i], zeros(Float64, P)...) for i=1:P ]
    convert(FDType, x[1])

    vp_fd = convert(Array{Array{FDType, 1}, 1}, mp.vp[1])
    mp_fd = ModelParams(vp_fd, mp.pp)
end

function convert(FDType::Type{ForwardDiff.HessianNumber},
                 mp::ModelParams{Float64})
    x = mp.vp[1]
    P = length(x)
    FDType = ForwardDiff.HessianNumber{length(mp.vp[1]), Float64}

    fd_x = [ ForwardDiff.HessianNumber(x[i], zeros(Float64, P)...) for i=1:P ]
    convert(FDType, x[1])

    vp_fd = convert(Array{Array{FDType, 1}, 1}, mp.vp[1])
    mp_fd = ModelParams(vp_fd, mp.pp)
end


# TODO: Maybe write it as a convert()?
function forward_diff_model_params{T <: Number}(
    FDType::Type{T},
    base_mp::ModelParams{Float64})
  S = length(base_mp.vp)
  P = length(base_mp.vp[1])
  mp_fd = ModelParams{FDType}([ zeros(FDType, P) for s=1:S ], base_mp.pp);
  # Set the values (but not gradient numbers) for parameters other
  # than the galaxy parameters.
  for s=1:base_mp.S, i=1:length(ids)
    mp_fd.vp[s][i] = base_mp.vp[s][i]
  end
  mp_fd.patches = base_mp.patches;
  mp_fd.tile_sources = base_mp.tile_sources;
  mp_fd.active_sources = base_mp.active_sources;
  mp_fd.objids = base_mp.objids;
  mp_fd
end


"""
Display model parameters with the variable names.
"""
function print_params(mp::ModelParams)
    for s in mp.active_sources
        println("=======================\n Object $(s):")
        for var_name in fieldnames(ids)
            println(var_name)
            println(mp.vp[s][ids.(var_name)])
        end
    end
end

"""
Display several model parameters side by side.
"""
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


"""
Display a Celeste catalog entry.
"""
function print_cat_entry(cat_entry::CatalogEntry)
    [println("$name: $(cat_entry.(name))") for name in
            fieldnames(cat_entry)]
end

#########################################################

# TODO: wrap this into its own module?
include(joinpath(Pkg.dir("Celeste"), "src/SensitiveFloat.jl"))

end
