module Model

# parameter types
export CatalogEntry,
       Image, Blob, TiledImage, TiledBlob, ImageTile, 
       SkyPatch, PsfComponent,
       GalaxyComponent, GalaxyPrototype,
       PriorParams, UnconstrainedParams,
       CanonicalParams, BrightnessParams, StarPosParams,
       GalaxyPosParams, GalaxyShapeParams,
       VariationalParams, FreeVariationalParams,
       PsfParams, RawPSF, CatalogEntry,
       init_source

# functions
export align

# constants
export band_letters, D, Ia, B, psf_K,
       galaxy_prototypes, prior,
       shape_standard_alignment,
       brightness_standard_alignment,
       gal_shape_alignment,
       ids_names, ids_free_names,
       ids, ids_free, star_ids, gal_ids, gal_shape_ids, psf_ids, bids


import Base.convert
import Base.+
import Distributions
import FITSIO
import WCS.WCSTransform
import ForwardDiff
import Logging

import Base.length

const cfgdir = joinpath(Pkg.dir("Celeste"), "cfg")

const band_letters = ['u', 'g', 'r', 'i', 'z']

# The number of bands (colors).
const B = length(band_letters)


include("light_source_model.jl")
include("psf_model.jl")
include("image_model.jl")
include("param_set.jl")
include("imaged_sources.jl")


# A vector of variational parameters.  The outer index is
# of celestial objects, and the inner index is over individual
# parameters for that object (referenced using ParamIndex).

typealias VariationalParams{NumType <: Number} Vector{Vector{NumType}}
typealias FreeVariationalParams{NumType <: Number} Vector{Vector{NumType}}

end  # module
