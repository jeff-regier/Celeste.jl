__precompile__()

module Celeste

# submodules
include("Configs.jl")
include("Log.jl")

include("SensitiveFloats.jl")

include("Model.jl")
include("Infer.jl")
include("ConstraintTransforms.jl")
include("Transform.jl")
include("PSF.jl")
include("SDSSIO.jl")

include("MCMC.jl")
include("StochasticVI.jl")
include("DeterministicVI.jl")
include("DeterministicVIImagePSF.jl")

include("ParallelRun.jl")

include("AccuracyBenchmark.jl")
include("GalsimBenchmark.jl")
include("Stripe82Score.jl")

include("CelesteEDA.jl")

include("ArgumentParse.jl")


import .ParallelRun: BoundingBox, OptimizedSource
import .SDSSIO: RunCamcolField, CatalogEntry
export BoundingBox, OptimizedSource, RunCamcolField, CatalogEntry

end # module
