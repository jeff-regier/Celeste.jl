module Celeste

# submodules
include("WCSUtils.jl")
include("Model.jl")
include("SensitiveFloats.jl")
include("ElboDeriv.jl")
include("Transform.jl")
include("PSF.jl")
include("SDSSIO.jl")
include("OptimizeElbo.jl")
include("ModelInit.jl")

# public API
export infer, score_field
include("api.jl")
include("score.jl")

end # module
