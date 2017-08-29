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
       ParamSet, SkyIntensity

# functions
export align

# constants
export NUM_COLOR_COMPONENTS, NUM_SOURCE_TYPES, NUM_BANDS,
       BAND_LETTERS, galaxy_prototypes, prior,
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
import ..Celeste: Const, @aliasscope, @unroll_loop
import ..SensitiveFloats: SensitiveFloat, clear!

import Base.length

const cfgdir = joinpath(Pkg.dir("Celeste"), "cfg")

const BAND_LETTERS = "ugriz"

# The number of bands (colors).
const NUM_BANDS = length(BAND_LETTERS)


include("model/light_source_model.jl")
include("model/psf_model.jl")
include("model/image_model.jl")
include("model/param_set.jl")
include("model/imaged_sources.jl")
include("model/wcs_utils.jl")
include("model/bivariate_normals.jl")
include("model/fsm_util.jl")
include("model/log_prob.jl")

end  # module
