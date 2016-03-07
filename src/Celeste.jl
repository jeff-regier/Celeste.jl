module Celeste

# submodules
include("Types.jl")
include("SensitiveFloats.jl")
include("Util.jl")
include("KL.jl")
include("ElboDeriv.jl")
include("SkyImages.jl")
include("Transform.jl")
include("OptimizeElbo.jl")
include("ModelInit.jl")
include("Synthetic.jl")
include("SampleData.jl")

# public API
export infer, score_field
include("api.jl")

end # module
