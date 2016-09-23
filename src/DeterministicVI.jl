"""
Calculate value, gradient, and hessian of the variational ELBO.
"""
module DeterministicVI

using ..Model
using ..SensitiveFloats
import ..SensitiveFloats.clear!
import ..Log
using ..Transform
import DataFrames
import Optim

export ElboArgs


"""
ElboArgs stores the arguments needed to evaluate the variational objective
function.
"""
type ElboArgs{NumType <: Number}
    S::Int64
    N::Int64

    # The number of components in the point spread function.
    psf_K::Int64
    images::Vector{TiledImage}
    vp::VariationalParams{NumType}
    tile_source_map::Vector{Matrix{Vector{Int}}}
    patches::Matrix{SkyPatch}
    active_sources::Vector{Int}

    # The number of standard deviations away below which a bivariate normal
    # will be calculated.  Set to Inf to evaluate bivariate normals at all points.
    num_allowed_sd::Float64
end


function ElboArgs{NumType <: Number}(
            images::Vector{TiledImage},
            vp::VariationalParams{NumType},
            tile_source_map::Vector{Matrix{Vector{Int}}},
            patches::Matrix{SkyPatch},
            active_sources::Vector{Int},
            psf_K::Int,
            num_allowed_sd::Float64)
    N = length(images)
    S = length(vp)

    @assert psf_K > 0
    @assert length(tile_source_map) == N
    @assert size(patches, 1) == S
    @assert size(patches, 2) == N
    ElboArgs(S, N, default_psf_K, images, vp, tile_source_map, patches,
             active_sources, num_allowed_sd)
end


include("deterministic_vi/elbo_kl.jl")
include("deterministic_vi/source_brightness.jl")
include("bivariate_normals.jl")
include("deterministic_vi/elbo.jl")
include("deterministic_vi/maximize_elbo.jl")


end
