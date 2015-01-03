module Celeste

require(joinpath(Pkg.dir("Celeste"),"src", "Util.jl"))
require(joinpath(Pkg.dir("Celeste"),"src", "CelesteTypes.jl"))
require(joinpath(Pkg.dir("Celeste"),"src", "ModelInit.jl"))
require(joinpath(Pkg.dir("Celeste"),"src", "StampBlob.jl"))
require(joinpath(Pkg.dir("Celeste"),"src", "Planck.jl"))
require(joinpath(Pkg.dir("Celeste"),"src", "ElboDeriv.jl"))
require(joinpath(Pkg.dir("Celeste"),"src", "OptimizeElbo.jl"))
require(joinpath(Pkg.dir("Celeste"),"src", "Synthetic.jl"))

import StampBlob
import Planck
import ElboDeriv
import OptimizeElbo
import ModelInit
import Synthetic

using CelesteTypes


# package code goes here

end # module
