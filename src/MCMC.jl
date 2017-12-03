module MCMC

using Celeste: Model, Transform, SensitiveFloats
import Celeste.Model: ImagePatch, Image
import Celeste.SDSSIO: SDSSBackground
using Distributions, StatsBase, DataFrames, StaticArrays, WCS
import Celeste: Config, Log

# TODO move these to model/log_prob.jl
star_param_names = ["lnr", "cug", "cgr", "cri", "ciz", "ra", "dec"]
galaxy_param_names = [star_param_names; ["gdev", "gaxis", "gangle", "gscale"]]

# functions to create star/gal log likelihoods
include("mcmc/mcmc_functions.jl")

# slice sampling and annealed importance sampling implementations
include("mcmc/slicesample.jl")
include("mcmc/ais.jl")

# misc MCMC functions
include("mcmc/mcmc_misc.jl")

# functions to run star/gal AIS, and combine inferences
include("mcmc/mcmc_infer.jl")

end
