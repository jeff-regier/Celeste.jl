
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

using Celeste: Param, @concretize, @inline_unnamed, ParameterizedArray

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
    e_axis::Param{:GalaxyShapeParams, :e_axis, ()}
    e_angle::Param{:GalaxyShapeParams, :e_angle, ()}
    e_scale::Param{:GalaxyShapeParams, :e_scale, ()}
end
@eval @concretize $GalaxyShapeParams
const gal_shape_ids = GalaxyShapeParams()

@inline_unnamed struct GalaxyPosParams <: ParamSet
    ::SharedPosParams
    e_dev::Param{:GalaxyPosParams, :e_dev, ()}
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
const bids = BrightnessParams{Any}()

@inline_unnamed immutable CanonicalParams <: ParamSet
    # shared u and galaxy_pos_params
    ::GalaxyPosParams
    
    star_brightness::BrightnessParams{Star}
    star_k::Param{CanonicalParams, :star_k, (D,)}
    
    galaxy_brightness::BrightnessParams{Star}
    galaxy_k::Param{CanonicalParams, :star_k, (D,)}
    
    a::Param{CanonicalParams, :a, (2,)}
end
@eval @concretize $CanonicalParams
const ids = CanonicalParams()

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
