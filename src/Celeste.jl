__precompile__()

module Celeste

# submodules
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
include("DeterministicVIImagePSF.jl")

include("ParallelRun.jl")

include("GalsimBenchmark.jl")
include("Stripe82Score.jl")

include("CelesteEDA.jl")

end # module
