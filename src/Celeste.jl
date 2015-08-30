module Celeste

require(joinpath(Pkg.dir("Celeste"), "src", "Util.jl"))
require(joinpath(Pkg.dir("Celeste"), "src", "Polygons.jl"))
require(joinpath(Pkg.dir("Celeste"), "src", "KL.jl"))
require(joinpath(Pkg.dir("Celeste"), "src", "CelesteTypes.jl"))
require(joinpath(Pkg.dir("Celeste"), "src", "Transform.jl"))
require(joinpath(Pkg.dir("Celeste"), "src", "Images.jl"))
require(joinpath(Pkg.dir("Celeste"), "src", "ModelInit.jl"))
require(joinpath(Pkg.dir("Celeste"), "src", "ElboDeriv.jl"))
require(joinpath(Pkg.dir("Celeste"), "src", "OptimizeElbo.jl"))
require(joinpath(Pkg.dir("Celeste"), "src", "Synthetic.jl"))
require(joinpath(Pkg.dir("Celeste"), "src", "SampleData.jl"))

import Polygons
import Images
import ElboDeriv
import OptimizeElbo
import ModelInit
import SampleData
import Transform
import KL

using CelesteTypes

end # module
