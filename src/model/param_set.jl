
# ParamSet types
#
# For more detailed explanations, see
# https://github.com/jeff-regier/Celeste.jl/wiki/Glossary-and-guide-to-Celeste-parameters
#
# The variable names are:
# u       = Location in world coordinates (formerly mu). In Celeste, world coordinates are generally
#           in degrees, but this depends on the WCS embedded in the image.
# e_dev   = Weight given to a galaxy of type 1 (formerly theta)
# e_axis  = Galaxy minor/major ratio (formerly rho)
# e_angle = Galaxy angle (formerly phi), in radians north of east
# e_scale = Galaxy scale (sigma). e_scale times sqrt(e_axis) gives the half-light radius in pixel
#           coords.
# For r1, r2, c1 and c2, the first row is stars, and the second is galaxies.
# r1      = Iax1 lognormal mean parameter for r_s, the brightness/total flux of the object, in nMgy.
#           For example, r1[1] gives the lognormal mean brightness for a star.
# r2      = Iax1 lognormal variance parameter for r_s.
# c1      = C_s means (formerly beta), the log ratios of brightness from each color band to the
#           previous one. For example c1[1,2] gives the mean log brightness ratio of band 3 over
#           band 2.
# c2      = C_s variances (formerly lambda)
# a       = probability of being a star or galaxy.  a[1, 1] is the
#           probability of being a star and a[2, 1] of being a galaxy.
#           (formerly chi)
# k       = {D|D-1}xIa matrix of color prior component indicators.
#           (formerly kappa)
#
# Note Ia denotes the number of types of astronomical objects (e.g., 2 for stars and galaxies).

using Celeste: Param, @concretize, @inline_unnamed, ParameterizedArray, ParamBatch, @define_accessors, Celeste, @create_sparse_implementation,
  diagonal_block, off_diagonal_block
import Celeste: to_batch
import Celeste: zero!

@compat abstract type ParamSet end

immutable Star; end
immutable Galaxy; end

type SharedPosParams <: ParamSet
    u::Param{SharedPosParams, :u, (2,)}
end
const StarPosParams = SharedPosParams
@eval @concretize $SharedPosParams
const star_ids = StarPosParams()
 
type GalaxyShapeParams <: ParamSet
    e_axis::Param{GalaxyShapeParams, :e_axis, ()}
    e_angle::Param{GalaxyShapeParams, :e_angle, ()}
    e_scale::Param{GalaxyShapeParams, :e_scale, ()}
end
@eval @concretize $GalaxyShapeParams
const gal_shape_ids = GalaxyShapeParams()

@inline_unnamed struct GalaxyPosParams <: ParamSet
    ::SharedPosParams
    e_dev::Param{GalaxyPosParams, :e_dev, ()}
    ::GalaxyShapeParams
end
@eval @concretize $GalaxyPosParams
const gal_ids = GalaxyPosParams()

type BrightnessParams{kind} <: ParamSet
    r1::Param{Tuple{BrightnessParams, kind}, :r1, ()}
    r2::Param{Tuple{BrightnessParams, kind}, :r2, ()}
    c1::Param{Tuple{BrightnessParams, kind}, :c1, (B-1,)}
    c2::Param{Tuple{BrightnessParams, kind}, :c2, (B-1,)}
end
@eval @concretize $BrightnessParams
const star_bids = BrightnessParams{Star}()
const gal_bids = BrightnessParams{Galaxy}()
bids(kind) = kind == Star() ? star_bids : gal_bids

#= This is defined this way for legacy reasons. CanonicalParams2 is preferred =#
immutable CanonicalParams <: ParamSet
    u::Vector{Int}
    e_dev::Int
    e_axis::Int
    e_angle::Int
    e_scale::Int
    r1::Vector{Int}
    r2::Vector{Int}
    c1::Matrix{Int}
    c2::Matrix{Int}
    a::Vector{Int}
    k::Matrix{Int}
    function CanonicalParams()
        new([1, 2], # u
            3, # e_dev
            4, # e_axis
            5, # e_angle
            6, # e_scale
            collect(7:(7+Ia-1)),  # r1
            collect((7+Ia):(7+2Ia-1)), # r2
            reshape((7+2Ia):(7+2Ia+(B-1)*Ia-1), (B-1, Ia)),  # c1
            reshape((7+2Ia+(B-1)*Ia):(7+2Ia+2*(B-1)*Ia-1), (B-1, Ia)),  # c2
            collect((7+2Ia+2*(B-1)*Ia):(7+3Ia+2*(B-1)*Ia-1)),  # a
            reshape((7+3Ia+2*(B-1)*Ia):(7+3Ia+2*(B-1)*Ia+D*Ia-1), (D, Ia))) # k
    end
end

const ids = CanonicalParams()
getids(::Type{CanonicalParams}) = ids
length(::Type{CanonicalParams}) = 6 + 3*Ia + 2*(B-1)*Ia + D*Ia
bright_ids(i) = [ids.r1[i]; ids.r2[i]; ids.c1[:, i]; ids.c2[:, i]]

@inline_unnamed immutable CanonicalParams2 <: ParamSet
    # shared u and galaxy_pos_params
    ::GalaxyPosParams

    star_brightness::BrightnessParams{Star}
    star_k::Param{CanonicalParams2, :star_k, (D,)}
    star_a::Param{CanonicalParams2, :star_a, ()}

    galaxy_brightness::BrightnessParams{Galaxy}
    galaxy_k::Param{CanonicalParams2, :galaxy_k, (D,)}
    galaxy_a::Param{CanonicalParams2, :galaxy_a, ()}
end

@eval @concretize $CanonicalParams2
const ids2 = CanonicalParams2()

const ids_2_to_ids = ParamBatch(tuple(ids2.u, ids2.e_dev, ids2.e_axis, ids2.e_angle, ids2.e_scale,
  ids2.star_brightness.r1, ids2.galaxy_brightness.r1,
  ids2.star_brightness.r2, ids2.galaxy_brightness.r2,
  ids2.star_brightness.c1, ids2.galaxy_brightness.c1,
  ids2.star_brightness.c2, ids2.galaxy_brightness.c2,
  ids2.star_a, ids2.galaxy_a,
  ids2.star_k, ids2.galaxy_k
))


a_param(::Star) = ids2.star_a
a_param(::Galaxy) = ids2.galaxy_a
non_u_shape_params(::Star) = ParamBatch(())
non_u_shape_params(::Galaxy) = ParamBatch((ids2.e_dev, ids2.e_axis, ids2.e_angle, ids2.e_scale))
shape_params(::Star) = ids2.u
shape_params(::Galaxy) = gal_ids
brightness_params(::Star) = ids2.star_brightness
brightness_params(::Galaxy) = ids2.galaxy_brightness

dense_diagonal_blocks(kind) = tuple(
  (brightness_params(kind), brightness_params(kind)),
  (length(non_u_shape_params(kind)) > 0 ? 
    ((non_u_shape_params(kind), non_u_shape_params(kind)),) :
    ())...
)
dense_off_diagonal_blocks(kind) = tuple(
  (brightness_params(kind), a_param(kind)),
  (shape_params(kind), a_param(kind)),
  (length(non_u_shape_params(kind)) > 0 ? 
      ((non_u_shape_params(kind), ids2.u),) :
      ())...,
  (brightness_params(kind), shape_params(kind))
)
all_dense_off_diagonal_blocks(kind) = tuple(
  dense_off_diagonal_blocks(kind)...,
  map(reverse, dense_off_diagonal_blocks(kind))...
)

const dense_blocks = tuple(
  (ids2.u, ids2.u),
  dense_diagonal_blocks(Star())...,
  dense_diagonal_blocks(Galaxy())...,
  all_dense_off_diagonal_blocks(Star())...,
  all_dense_off_diagonal_blocks(Galaxy())...,  
)

const symmetric_dense_blocks = tuple(
  (ids2.u, ids2.u),
  dense_diagonal_blocks(Star())...,
  dense_diagonal_blocks(Galaxy())...,
  dense_off_diagonal_blocks(Star())...,
  dense_off_diagonal_blocks(Galaxy())...,  
)

const dense_block_mapping = map(x->(ids_2_to_ids[x[1]],ids_2_to_ids[x[2]]), dense_blocks)
const symmetric_dense_block_mapping = 
  map(x->(ids_2_to_ids[x[1]],ids_2_to_ids[x[2]]), symmetric_dense_blocks)

@eval @define_accessors $CanonicalParams2 ids2 mutable struct SparseStruct{NumType}
    u_u_block::diagonal_block(NumType, ids2.u)
    star_bright_bright_block::diagonal_block(NumType, brightness_params(Star()))
    gal_shape_shape_block::diagonal_block(NumType, non_u_shape_params(Galaxy()))
    gal_bright_bright_block::diagonal_block(NumType, brightness_params(Galaxy()))  
    star_bright_a_block::off_diagonal_block(NumType, brightness_params(Star()), a_param(Star()))
    star_shape_a_block::off_diagonal_block(NumType, shape_params(Star()), a_param(Star()))
    star_bright_shape_block::off_diagonal_block(NumType, brightness_params(Star()), shape_params(Star()))
    gal_bright_a_block::off_diagonal_block(NumType, brightness_params(Galaxy()), a_param(Galaxy()))
    gal_shape_a_block::off_diagonal_block(NumType, shape_params(Galaxy()), a_param(Galaxy()))
    gal_shape_u_block::off_diagonal_block(NumType, non_u_shape_params(Galaxy()), ids2.u)
    gal_bright_shape_block::off_diagonal_block(NumType, brightness_params(Galaxy()), shape_params(Galaxy()))
end
@eval @create_sparse_implementation $SparseStruct SparseStruct
Celeste.is_implicitly_symmetric(s::SparseStruct) = true
Base.issymmetric(s::SparseStruct) = true

type LatentStateIndexes <: ParamSet
    u::Vector{Int}
    e_dev::Int
    e_axis::Int
    e_angle::Int
    e_scale::Int
    r::Vector{Int}
    c::Matrix{Int}
    a::Matrix{Int}
    k::Matrix{Int}        # (not needed, i think)

    LatentStateIndexes() =
        new([1, 2], 3, 4, 5, 6,
            collect(7:(7+Ia-1)),  # r
            reshape((7+Ia):(7+Ia+(B-1)*Ia-1), (B-1, Ia)),  # c
            reshape((7+Ia+(B-1)*Ia):(7+2Ia+(B-1)*Ia-1), (Ia, 1)),  # a
            reshape((7+2Ia+(B-1)*Ia):(7+2Ia+(B-1)*Ia+D*Ia-1), (D, Ia))) # k
end

const lidx = LatentStateIndexes()
getlidx(::Type{LatentStateIndexes}) = lidx
length(::Type{LatentStateIndexes}) = 22 #6 + 3*Ia + 2*(B-1)*Ia + D*Ia

# Parameters for a representation of the PSF
immutable PsfParams <: ParamSet
    mu::UnitRange{Int}
    e_axis::Int
    e_angle::Int
    e_scale::Int
    weight::Int

    function PsfParams()
      new(1:2, 3, 4, 5, 6)
    end
end
const psf_ids = PsfParams()
getids(::Type{PsfParams}) = psf_ids
length(::Type{PsfParams}) = 6

# define length(value) in addition to length(Type) for ParamSets
length{T<:ParamSet}(::T) = length(T)

#TODO: build these from ue_align, etc., here.
align(::StarPosParams, CanonicalParams) = ids.u
align(::GalaxyPosParams, CanonicalParams) =
   [ids.u; ids.e_dev; ids.e_axis; ids.e_angle; ids.e_scale]
align(::CanonicalParams, CanonicalParams) = collect(1:length(CanonicalParams))
align(::GalaxyShapeParams, GalaxyPosParams) =
  [gal_ids.e_axis; gal_ids.e_angle; gal_ids.e_scale]

# The shape and brightness parameters for stars and galaxies respectively.
const shape_standard_alignment = (ids.u,
   [ids.u; ids.e_dev; ids.e_axis; ids.e_angle; ids.e_scale])
bright_ids(i) = [ids.r1[i]; ids.r2[i]; ids.c1[:, i]; ids.c2[:, i]]
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
