# A vector of variational parameters.  The outer index is
# of celestial objects, and the inner index is over individual
# parameters for that object (referenced using ParamIndex).

typealias VariationalParams{NumType <: Number} Vector{Vector{NumType}}
typealias RectVariationalParams{NumType <: Number} Vector{Vector{NumType}}
typealias FreeVariationalParams{NumType <: Number} Vector{Vector{NumType}}


"""
The parameters for a particular image.

Attributes:
 - vp: The variational parameters
 - pp: The prior parameters
 - patches: An (objects X bands) matrix of SkyPatch objects
 - tile_width: The number of pixels across a tile
 - tile_sources: A vector (over bands) of an array (over tiles) of vectors
                 of sources influencing each tile.
 - active_sources: Indices of the sources that are currently being fit by the
                   model.
 - objids: Global object ids for the sources in this ModelParams object.

 - S: The number of sources.
"""
type ModelParams{T <: Number}
    vp::VariationalParams{T}
    pp::PriorParams
    patches::Array{SkyPatch, 2}
    tile_sources::Vector{Array{Vector{Int}, 2}}
    active_sources::Vector{Int}
    objids::Vector{ASCIIString}

    S::Int

    function ModelParams(vp::VariationalParams{T}, pp::PriorParams)
        # There must be one patch for each celestial object.
        S = length(vp)
        all_tile_sources = fill(fill(collect(1:S), 1, 1), 5)
        patches = Array(SkyPatch, S, 5)
        active_sources = collect(1:S)
        objids = ASCIIString[string(s) for s in 1:S]

        new(vp, pp, patches, all_tile_sources, active_sources, objids, S)
    end
end


# Make a copy of a ModelParams keeping only some sources.
function ModelParams{T <: Number}(mp_all::ModelParams{T}, keep_s::Vector{Int})
    mp = ModelParams{T}(deepcopy(mp_all.vp[keep_s]), mp_all.pp);
    mp.active_sources = Int[]
    mp.objids = Array(ASCIIString, length(keep_s))
    mp.patches = Array(SkyPatch, mp.S, size(mp_all.patches, 2))

    # Indices of sources in the new model params
    for sa in 1:length(keep_s)
        s = keep_s[sa]
        mp.objids[sa] = mp_all.objids[s]
        mp.patches[sa, :] = mp_all.patches[s, :]
        if s in mp_all.active_sources
            push!(mp.active_sources, sa)
        end
    end

    @assert length(mp_all.tile_sources) == size(mp_all.patches, 2)
    num_bands = length(mp_all.tile_sources)
    mp.tile_sources = Array(Matrix{Vector{Int}}, num_bands)
    for b=1:num_bands
        mp.tile_sources[b] = Array(Vector{Int}, size(mp_all.tile_sources[b]))
        for tile_ind in 1:length(mp_all.tile_sources[b])
                tile_s = intersect(mp_all.tile_sources[b][tile_ind], keep_s)
                mp.tile_sources[b][tile_ind] =
                    Int[ findfirst(keep_s, s) for s in tile_s ]
        end
    end

    mp
end

ModelParams{T <: Number}(vp::VariationalParams{T}, pp::PriorParams) =
    ModelParams{T}(vp, pp)
