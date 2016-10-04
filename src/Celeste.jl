module Celeste

# submodules
include("Exceptions.jl")
include("Log.jl")

include("SensitiveFloats.jl")

include("Model.jl")
include("Transform.jl")
include("PSF.jl")
include("SDSSIO.jl")

include("DeterministicVI.jl")
include("StochasticVI.jl")
include("MCMC.jl")

include("Infer.jl")
include("ParallelRun.jl")

include("Stripe82Score.jl")

end # module
