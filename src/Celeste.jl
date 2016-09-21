module Celeste

# submodules
include("Log.jl")

include("SensitiveFloats.jl")

include("Model.jl")
include("DeterministicVI.jl")
include("Transform.jl")
include("PSF.jl")
include("SDSSIO.jl")
include("OptimizeElbo.jl")
include("Infer.jl")
include("ParallelRun.jl")

include("Stripe82Score.jl")

end # module
