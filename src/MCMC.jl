module MCMC

using Celeste: Model, Transform, SensitiveFloats, MCMC
using Distributions, StatsBase, DataFrames, StaticArrays, WCS

# TODO move these to model/log_prob.jl
star_param_names = ["lnr", "cug", "cgr", "cri", "ciz", "ra", "dec"]
galaxy_param_names = [star_param_names; ["gdev", "gaxis", "gangle", "gscale"]]

# functions to create star/gal log likelihoods
include("mcmc/mcmc_functions.jl")

# slice sampling function
include("mcmc/slicesample.jl")
include("mcmc/ais.jl")

# misc MCMC functions
include("mcmc/mcmc_misc.jl")

end
