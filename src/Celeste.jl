module Celeste

# submodules
include("Log.jl")
include("WCSUtils.jl")
include("Model.jl")
include("SensitiveFloats.jl")
include("ElboDeriv.jl")
include("Transform.jl")
include("PSF.jl")
include("SDSSIO.jl")
include("OptimizeElbo.jl")
include("Infer.jl")

# public API
export infer, score_field
include("api.jl")
include("nersc_io.jl")
include("score.jl")

end # module
