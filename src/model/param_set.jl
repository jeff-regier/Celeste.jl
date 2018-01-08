
# ParamSet types
#
# For more detailed explanations, see
# https://github.com/jeff-regier/Celeste.jl/wiki/Glossary-and-guide-to-Celeste-parameters
#
# The variable names are:
# pos       = Location in world coordinates. Right ascension and declination
# gal_frac_dev   = Weight given to a galaxy of type 1 (formerly theta)
# gal_axis_ratio  = Galaxy minor/major ratio (formerly rho)
# gal_angle = Galaxy angle (formerly phi), in radians north of east
# gal_radius_px = Galaxy scale (sigma). gal_radius_px times sqrt(gal_axis_ratio) gives the half-light radius in pixel
#           coords.
# For flux_loc, flux_scale, color_mean and color_var, the first row is stars, and the second is galaxies.
# flux_loc      = NUM_SOURCE_TYPESx1 lognormal mean parameter for r_s, the brightness/total flux of the object, in nMgy.
#           For example, flux_loc[1] gives the lognormal mean brightness for a star.
# flux_scale      = NUM_SOURCE_TYPESx1 lognormal variance parameter for r_s.
# color_mean      = C_s means (formerly beta), the log ratios of brightness from each color band to the
#           previous one. For example color_mean[1,2] gives the mean log brightness ratio of band 3 over
#           band 2.
# color_var      = C_s variances (formerly lambda)
# is_star       = probability of being a star or galaxy.  is_star[1, 1] is the
#           probability of being a star and is_star[2, 1] of being a galaxy.
#           (formerly chi)
# k       = {NUM_COLOR_COMPONENTS|NUM_COLOR_COMPONENTS-1}xNUM_SOURCE_TYPES matrix of color prior component indicators.
#           (formerly kappa)
#
# Note NUM_SOURCE_TYPES denotes the number of types of astronomical objects (e.g., 2 for stars and galaxies).

abstract type ParamSet end

struct StarPosParams <: ParamSet
    pos::Vector{Int}
    StarPosParams() = new([1, 2])
end
const star_ids = StarPosParams()
getids(::Type{StarPosParams}) = star_ids
length(::Type{StarPosParams}) = 2

struct GalaxyShapeParams <: ParamSet
    gal_axis_ratio::Int
    gal_angle::Int
    gal_radius_px::Int
    GalaxyShapeParams() = new(1, 2, 3)
end
const gal_shape_ids = GalaxyShapeParams()
getids(::Type{GalaxyShapeParams}) = gal_shape_ids
length(::Type{GalaxyShapeParams}) = 3

struct GalaxyPosParams <: ParamSet
    pos::Vector{Int}
    gal_frac_dev::Int
    gal_axis_ratio::Int
    gal_angle::Int
    gal_radius_px::Int
    GalaxyPosParams() = new([1, 2], 3, 4, 5, 6)
end
const gal_ids = GalaxyPosParams()
getids(::Type{GalaxyPosParams}) = gal_ids
length(::Type{GalaxyPosParams}) = 6


struct BrightnessParams <: ParamSet
    flux_loc::Int
    flux_scale::Int
    color_mean::Vector{Int}
    color_var::Vector{Int}
    BrightnessParams() = new(1, 2,
                             collect(3:(3+(NUM_BANDS-1)-1)),
                             collect((3+NUM_BANDS-1):(3+2*(NUM_BANDS-1)-1)))
end
const bids = BrightnessParams()
getids(::Type{BrightnessParams}) = bids
length(::Type{BrightnessParams}) = 2 + 2 * (NUM_BANDS-1)

struct CanonicalParams <: ParamSet
    pos::Vector{Int}
    gal_frac_dev::Int
    gal_axis_ratio::Int
    gal_angle::Int
    gal_radius_px::Int
    flux_loc::Vector{Int}
    flux_scale::Vector{Int}
    color_mean::Matrix{Int}
    color_var::Matrix{Int}
    is_star::Vector{Int}
    k::Matrix{Int}
    function CanonicalParams()
        new([1, 2], # pos
            3, # gal_frac_dev
            4, # gal_axis_ratio
            5, # gal_angle
            6, # gal_radius_px
            collect(7:(7+NUM_SOURCE_TYPES-1)),  # flux_loc
            collect((7+NUM_SOURCE_TYPES):(7+2NUM_SOURCE_TYPES-1)), # flux_scale
            reshape((7+2NUM_SOURCE_TYPES):(7+2NUM_SOURCE_TYPES+(NUM_BANDS-1)*NUM_SOURCE_TYPES-1), (NUM_BANDS-1, NUM_SOURCE_TYPES)),  # color_mean
            reshape((7+2NUM_SOURCE_TYPES+(NUM_BANDS-1)*NUM_SOURCE_TYPES):(7+2NUM_SOURCE_TYPES+2*(NUM_BANDS-1)*NUM_SOURCE_TYPES-1), (NUM_BANDS-1, NUM_SOURCE_TYPES)),  # color_var
            collect((7+2NUM_SOURCE_TYPES+2*(NUM_BANDS-1)*NUM_SOURCE_TYPES):(7+3NUM_SOURCE_TYPES+2*(NUM_BANDS-1)*NUM_SOURCE_TYPES-1)),  # is_star
            reshape((7+3NUM_SOURCE_TYPES+2*(NUM_BANDS-1)*NUM_SOURCE_TYPES):(7+3NUM_SOURCE_TYPES+2*(NUM_BANDS-1)*NUM_SOURCE_TYPES+NUM_COLOR_COMPONENTS*NUM_SOURCE_TYPES-1), (NUM_COLOR_COMPONENTS, NUM_SOURCE_TYPES))) # k
    end
end

const ids = CanonicalParams()

getids(::Type{CanonicalParams}) = ids

length(::Type{CanonicalParams}) = 6 + 3*NUM_SOURCE_TYPES + 2*(NUM_BANDS-1)*NUM_SOURCE_TYPES + NUM_COLOR_COMPONENTS*NUM_SOURCE_TYPES


struct LatentStateIndexes <: ParamSet
    pos::Vector{Int}
    gal_frac_dev::Int
    gal_axis_ratio::Int
    gal_angle::Int
    gal_radius_px::Int
    flux::Vector{Int}
    color::Matrix{Int}
    is_star::Matrix{Int}
    k::Matrix{Int}        # (not needed, i think)

    LatentStateIndexes() =
        new([1, 2], 3, 4, 5, 6,
            collect(7:(7+NUM_SOURCE_TYPES-1)),  # r
            reshape((7+NUM_SOURCE_TYPES):(7+NUM_SOURCE_TYPES+(NUM_BANDS-1)*NUM_SOURCE_TYPES-1), (NUM_BANDS-1, NUM_SOURCE_TYPES)),  # c
            reshape((7+NUM_SOURCE_TYPES+(NUM_BANDS-1)*NUM_SOURCE_TYPES):(7+2NUM_SOURCE_TYPES+(NUM_BANDS-1)*NUM_SOURCE_TYPES-1), (NUM_SOURCE_TYPES, 1)),  # is_star
            reshape((7+2NUM_SOURCE_TYPES+(NUM_BANDS-1)*NUM_SOURCE_TYPES):(7+2NUM_SOURCE_TYPES+(NUM_BANDS-1)*NUM_SOURCE_TYPES+NUM_COLOR_COMPONENTS*NUM_SOURCE_TYPES-1), (NUM_COLOR_COMPONENTS, NUM_SOURCE_TYPES))) # k
end

const lidx = LatentStateIndexes()
getlidx(::Type{LatentStateIndexes}) = lidx
length(::Type{LatentStateIndexes}) = 22 #6 + 3*NUM_SOURCE_TYPES + 2*(NUM_BANDS-1)*NUM_SOURCE_TYPES + NUM_COLOR_COMPONENTS*NUM_SOURCE_TYPES

# Parameters for a representation of the PSF
struct PsfParams <: ParamSet
    mu::UnitRange{Int}
    gal_axis_ratio::Int
    gal_angle::Int
    gal_radius_px::Int
    weight::Int

    function PsfParams()
      new(1:2, 3, 4, 5, 6)
    end
end
const psf_ids = PsfParams()
getids(::Type{PsfParams}) = psf_ids
length(::Type{PsfParams}) = 6

# define length(value) in addition to length(Type) for ParamSets
length(::T) where {T<:ParamSet} = length(T)

#TODO: build these from ue_align, etc., here.
align(::StarPosParams, ids) = ids.pos
align(::GalaxyPosParams, ids) =
   [ids.pos; ids.gal_frac_dev; ids.gal_axis_ratio; ids.gal_angle; ids.gal_radius_px]
align(::CanonicalParams, _ids) = collect(1:length(CanonicalParams))
align(::GalaxyShapeParams, gal_ids) =
  [gal_ids.gal_axis_ratio; gal_ids.gal_angle; gal_ids.gal_radius_px]

# The shape and brightness parameters for stars and galaxies respectively.
const shape_standard_alignment = (ids.pos,
   [ids.pos; ids.gal_frac_dev; ids.gal_axis_ratio; ids.gal_angle; ids.gal_radius_px])
bright_ids(i) = [ids.flux_loc[i]; ids.flux_scale[i]; ids.color_mean[:, i]; ids.color_var[:, i]]
const brightness_standard_alignment = (bright_ids(1), bright_ids(2))

# Note that gal_shape_alignment aligns the shape ids with the GalaxyPosParams,
# not the CanonicalParams.
const gal_shape_alignment = Const(align(gal_shape_ids, gal_ids))

function get_id_names(ids::CanonicalParams)
    ids_names = Vector{String}(length(ids))
    for name in fieldnames(ids)
        inds = getfield(ids, name)
        if isa(inds, Matrix)
            for i in 1:size(inds, 1), j in 1:size(inds, 2)
                ids_names[inds[i, j]] = "$(name)_$(i)_$(j)"
            end
        elseif isa(inds, Vector)
            for i in eachindex(inds)
                ids_names[inds[i]] = "$(name)_$(i)"
            end
        elseif isa(inds, Int)
            ids_names[inds] = "$(name)_$(inds)"
        else
            error("found unsupported index type for parameter $(name): $(typeof(inds))")
        end
    end
    return ids_names
end

const ids_names = get_id_names(ids)
