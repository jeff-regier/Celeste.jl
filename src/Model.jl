module Model

using Compat
import ForwardDiff
using StaticArrays

# parameter types
export Image, SkyPatch, PsfComponent,
       GalaxyComponent, GalaxyPrototype,
       PriorParams, CanonicalParams, BrightnessParams, StarPosParams,
       GalaxyPosParams, GalaxyShapeParams,
       PsfParams, RawPSF, CatalogEntry,
       ParamSet

# functions
export align

# constants
export band_letters, D, Ia, B,
       galaxy_prototypes, prior,
       shape_standard_alignment,
       brightness_standard_alignment,
       gal_shape_alignment,
       ids_names,
       ids, star_ids, gal_ids, gal_shape_ids, psf_ids, bids


import Base.convert
import Base.+
import Distributions
import FITSIO, WCS
import WCS.WCSTransform
import ..Log

import Base.length

const cfgdir = joinpath(Pkg.dir("Celeste"), "cfg")

const band_letters = ['u', 'g', 'r', 'i', 'z']

# The number of bands (colors).
const B = length(band_letters)


include("model/light_source_model.jl")
include("model/psf_model.jl")
include("model/image_model.jl")
include("model/param_set.jl")
include("model/imaged_sources.jl")
include("model/wcs_utils.jl")

import ..SensitiveFloats: SensitiveFloat, clear!
include("bivariate_normals.jl")
include("model/fsm_util.jl")
include("model/log_prob.jl")

end  # module
