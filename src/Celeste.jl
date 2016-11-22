__precompile__()

module Celeste

# submodules
include("Log.jl")

include("SensitiveFloats.jl")

include("Model.jl")
include("Transform.jl")
include("PSF.jl")
include("SDSSIO.jl")

include("DeterministicVI.jl")
include("DeterministicVIImagePSF.jl")
include("StochasticVI.jl")
include("MCMC.jl")

include("Infer.jl")
include("ParallelRun.jl")

include("Stripe82Score.jl")

include("CelesteEDA.jl")

end # module
