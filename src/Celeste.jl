__precompile__()

module Celeste

# import this before StaticArrays has a chance to load, to
# workaround a bug
import ForwardDiff

# alias scopes
include("aliasscopes.jl")

# submodules
include("Configs.jl")
include("Log.jl")

include("SensitiveFloats.jl")

include("Model.jl")
include("Infer.jl")
include("Transform.jl")
include("PSF.jl")
include("SDSSIO.jl")

include("MCMC.jl")
include("StochasticVI.jl")
include("DeterministicVI.jl")

include("ParallelRun.jl")

include("Synthetic.jl")
include("AccuracyBenchmark.jl")
include("GalsimBenchmark.jl")

include("ArgumentParse.jl")


import .ParallelRun: BoundingBox, OptimizedSource
import .SDSSIO: RunCamcolField, CatalogEntry
export BoundingBox, OptimizedSource, RunCamcolField, CatalogEntry

end # module
