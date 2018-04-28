__precompile__()

module Celeste

# import this before StaticArrays has a chance to load, to workaround a bug
import ForwardDiff

include("aliasscopes.jl")
include("config.jl")
include("dataset.jl")

# submodules
include("Log.jl")

include("SensitiveFloats.jl")
include("BivariateNormals.jl")

include("Coordinates.jl")

include("Model.jl")
include("Transform.jl")
include("PSF.jl")
include("SDSSIO.jl")
include("SEP.jl")

include("detection.jl")
include("MCMC.jl")
include("StochasticVI.jl")
include("DeterministicVI.jl")
include("ParallelRun.jl")

include("Synthetic.jl")
include("AccuracyBenchmark.jl")
include("GalsimBenchmark.jl")

include("DECALSIO.jl")
include("ArgumentParse.jl")
include("main.jl")

import .ParallelRun: BoundingBox, OptimizedSource
import .SDSSIO: RunCamcolField, CatalogEntry
export BoundingBox, OptimizedSource, RunCamcolField, CatalogEntry

end # module
