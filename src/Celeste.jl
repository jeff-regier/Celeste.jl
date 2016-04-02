module Celeste

# submodules
include("SDSS.jl")
include("WCSUtils.jl")
include("SDSSIO.jl")
include("Types.jl")
include("SensitiveFloats.jl")
include("KL.jl")
include("BivariateNormals.jl")
include("Transform.jl")
include("PSF.jl")
include("ElboDeriv.jl")
include("SkyImages.jl")
include("OptimizeElbo.jl")
include("ModelInit.jl")
include("Synthetic.jl")
include("SampleData.jl")

# public API
export infer, score_field
include("api.jl")
include("score.jl")

end # module
