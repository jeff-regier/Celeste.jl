
# ParamSet types
#
# The variable names are:
# u       = Location in world coordinates (formerly mu)
# e_dev   = Weight given to a galaxy of type 1 (formerly theta)
# e_axis  = Galaxy minor/major ratio (formerly rho)
# e_angle = Galaxy angle (formerly phi)
# e_scale = Galaxy scale (sigma)
# For r1 and r2, the first row is stars, and the second is galaxies.
# r1      = Iax1 lognormal mean parameter for r_s.
# r2      = Iax1 lognormal variance parameter for r_s.
# c1      = C_s means (formerly beta)
# c2      = C_s variances (formerly lambda)
# a       = probability of being a star or galaxy.  a[1] is the
#           probability of being a star and a[2] of being a galaxy.
#           (formerly chi)
# k       = {D|D-1}xIa matrix of color prior component indicators.
#           (formerly kappa)

import ..SensitiveFloats: ParamSet

type StarPosParams <: ParamSet
    u::Vector{Int}
    StarPosParams() = new([1, 2])
end
const star_ids = StarPosParams()
getids(::Type{StarPosParams}) = star_ids
length(::Type{StarPosParams}) = 2


type GalaxyShapeParams <: ParamSet
    e_axis::Int
    e_angle::Int
    e_scale::Int
    GalaxyShapeParams() = new(1, 2, 3)
end
const gal_shape_ids = GalaxyShapeParams()
getids(::Type{GalaxyShapeParams}) = gal_shape_ids
length(::Type{GalaxyShapeParams}) = 3


type GalaxyPosParams <: ParamSet
    u::Vector{Int}
    e_dev::Int
    e_axis::Int
    e_angle::Int
    e_scale::Int
    GalaxyPosParams() = new([1, 2], 3, 4, 5, 6)
end
const gal_ids = GalaxyPosParams()
getids(::Type{GalaxyPosParams}) = gal_ids
length(::Type{GalaxyPosParams}) = 6


type BrightnessParams <: ParamSet
    r1::Int
    r2::Int
    c1::Vector{Int}
    c2::Vector{Int}
    BrightnessParams() = new(1, 2,
                             collect(3:(3+(B-1)-1)),
                             collect((3+B-1):(3+2*(B-1)-1)))
end
const bids = BrightnessParams()
getids(::Type{BrightnessParams}) = bids
length(::Type{BrightnessParams}) = 2 + 2 * (B-1)


type CanonicalParams <: ParamSet
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
    CanonicalParams() =
        new([1, 2], 3, 4, 5, 6,
            collect(7:(7+Ia-1)),  # r1
            collect((7+Ia):(7+2Ia-1)), # r2
            reshape((7+2Ia):(7+2Ia+(B-1)*Ia-1), (B-1, Ia)),  # c1
            reshape((7+2Ia+(B-1)*Ia):(7+2Ia+2*(B-1)*Ia-1), (B-1, Ia)),  # c2
            collect((7+2Ia+2*(B-1)*Ia):(7+3Ia+2*(B-1)*Ia-1)),  # a
            reshape((7+3Ia+2*(B-1)*Ia):(7+3Ia+2*(B-1)*Ia+D*Ia-1), (D, Ia))) # k
end
const ids = CanonicalParams()
getids(::Type{CanonicalParams}) = ids
length(::Type{CanonicalParams}) = 6 + 3*Ia + 2*(B-1)*Ia + D*Ia


type UnconstrainedParams <: ParamSet
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
    UnconstrainedParams() =
        new([1, 2], 3, 4, 5, 6,
            collect(7:(7+Ia-1)),  # r1
            collect((7+Ia):(7+2Ia-1)), # r2
            reshape((7+2Ia):(7+2Ia+(B-1)*Ia-1), (B-1, Ia)),  # c1
            reshape((7+2Ia+(B-1)*Ia):(7+2Ia+2*(B-1)*Ia-1), (B-1, Ia)),  # c2
            collect((7+2Ia+2*(B-1)*Ia):(7+2Ia+2*(B-1)*Ia+(Ia-1)-1)),  # a
            reshape((7+2Ia+2*(B-1)*Ia+(Ia-1)):
                    (7+2Ia+2*(B-1)*Ia+(Ia-1)+(D-1)*Ia-1), (D-1, Ia))) # k
end
const ids_free = UnconstrainedParams()
getids(::Type{UnconstrainedParams}) = ids_free
length(::Type{UnconstrainedParams}) =  6 + 2*Ia + 2*(B-1)*Ia + (D-1)*Ia + Ia-1

# Parameters for a representation of the PSF
type PsfParams <: ParamSet
    mu::Vector{Int}
    e_axis::Int
    e_angle::Int
    e_scale::Int
    weight::Int

    function PsfParams()
      new([1, 2], 3, 4, 5, 6)
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
const gal_shape_alignment = align(gal_shape_ids, gal_ids)

function get_id_names(
  ids::Union{CanonicalParams, UnconstrainedParams})
  ids_names = Array(String, length(ids))
  for name in fieldnames(ids)
    inds = getfield(ids, name)
    if length(size(inds)) == 0
      ids_names[inds] = "$(name)"
    elseif length(size(inds)) == 1
      for i = 1:size(inds)[1]
          ids_names[inds[i]] = "$(name)_$(i)"
      end
    elseif length(size(inds)) == 2
      for i = 1:size(inds)[1], j = 1:size(inds)[2]
          ids_names[inds[i, j]] = "$(name)_$(i)_$(j)"
      end
    else
      error("Names of 3d parameters not supported ($(name))")
    end
  end
  return ids_names
end

const ids_names = get_id_names(ids)
const ids_free_names = get_id_names(ids_free)



