module Celeste

require(joinpath(Pkg.dir("Celeste"), "src", "Util.jl"))
require(joinpath(Pkg.dir("Celeste"), "src", "CelesteTypes.jl"))
require(joinpath(Pkg.dir("Celeste"), "src", "ModelInit.jl"))
require(joinpath(Pkg.dir("Celeste"), "src", "SDSS.jl"))
require(joinpath(Pkg.dir("Celeste"), "src", "ElboDeriv.jl"))
require(joinpath(Pkg.dir("Celeste"), "src", "OptimizeElbo.jl"))
require(joinpath(Pkg.dir("Celeste"), "src", "Synthetic.jl"))
require(joinpath(Pkg.dir("Celeste"), "src", "SampleData.jl"))

import SDSS
import ElboDeriv
import OptimizeElbo
import ModelInit

using CelesteTypes


# package code goes here

end # module
