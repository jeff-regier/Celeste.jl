
using CelesteTypes.getids
using CelesteTypes.HessianEntry

export multiply_sf!

#########################################################

@doc """
Set a SensitiveFloat's hessian term, maintaining symmetry.
""" ->
function set_hess!{ParamType <: CelesteTypes.ParamSet, NumType <: Number}(
    sf::SensitiveFloat{ParamType, NumType},
    i::Int64, j::Int64, v::NumType)
  set_hess!(sf, i, j, 1, v)
end

function set_hess!{ParamType <: CelesteTypes.ParamSet, NumType <: Number}(
    sf::SensitiveFloat{ParamType, NumType},
    i::Int64, j::Int64, s::Int64, v::NumType)
  i != j ?
    sf.hs[s][i, j] = sf.hs[s][j, i] = v:
    sf.hs[s][i, j] = v
end

function zero_sensitive_float{ParamType <: CelesteTypes.ParamSet}(
  ::Type{ParamType}, NumType::DataType, local_S::Int64)
    local_P = length(ParamType)
    d = zeros(NumType, local_P, local_S)
    h = HessianEntry{NumType}[]

    # TODO: is there some kind of symmetric matrix type to use?
    hs = fill(zeros(local_P, local_P), local_S)
    SensitiveFloat{ParamType, NumType}(
      zero(NumType), d, h, hs, getids(ParamType))
end

function zero_sensitive_float{ParamType <: CelesteTypes.ParamSet}(
  ::Type{ParamType}, NumType::DataType)
    # Default to a single source.
    zero_sensitive_float(ParamType, NumType, 1)
end

function clear!{ParamType <: CelesteTypes.ParamSet, NumType <: Number}(
  sp::SensitiveFloat{ParamType, NumType})
    sp.v = zero(NumType)
    h = HessianEntry{NumType}[]
    fill!(sp.d, zero(NumType))
    [ fill!(sp.hs[s], zero(NumType)) for s=1:size(sp.d)[2] ]
end

# If no type is specified, default to using Float64.
function zero_sensitive_float{ParamType <: CelesteTypes.ParamSet}(
  param_arg::Type{ParamType}, local_S::Int64)
    zero_sensitive_float(param_arg, Float64, local_S)
end

function zero_sensitive_float{ParamType <: CelesteTypes.ParamSet}(param_arg::Type{ParamType})
    zero_sensitive_float(param_arg, Float64, 1)
end

function +(sf1::SensitiveFloat, sf2::SensitiveFloat)
  S = size(sf1.d)[2]

  # Simply asserting equality of the ids doesn't work for some reason.
  @assert typeof(sf1.ids) == typeof(sf2.ids)
  @assert length(sf1.ids) == length(sf2.ids)
  [ @assert size(sf1.hs[s]) == size(sf2.hs[s]) for s=1:S ]

  @assert size(sf1.d) == size(sf2.d)

  sf3 = deepcopy(sf1)
  sf3.v = sf1.v + sf2.v
  sf3.d = sf1.d + sf2.d
  sf3.h = vcat(sf1.h, sf2.h)
  for s=1:S
    sf3.hs = sf1.hs + sf2.hs
  end

  sf3
end


@doc """
Updates sf1 in place with sf1 * sf2.

ids1 and ids2 are the ids for which sf1 and sf2 have nonzero derivatives,
respectively.
""" ->
function multiply_sf!{ParamType <: CelesteTypes.ParamSet, NumType <: Number}(
    sf1::SensitiveFloat{ParamType, NumType},
    sf2::SensitiveFloat{ParamType, NumType};
    ids1::Array{Int64}=collect(1:length(sf1.ids)),
    ids2::Array{Int64}=collect(1:length(sf2.ids)))

  S = size(sf1.d)[2]

  # Can you actually do this in place without overwriting things you need?

  # Chain rule for second derivatives.
  for s=1:S
    #fill!(sf1.hs[s], 0.0)
    # Second derivate terms involving second derivates.  TODO: nearly half are redudnant.
    for i in ids1, j in ids1
      sf1.hs[s][i, j] = sf2.v * sf1.hs[s][i, j]
    end
    for i in ids2, j in ids2
      sf1.hs[s][i, j] += sf1.v * sf2.hs[s][i, j]
    end

    # Second derivative terms involving first derivatives.
    for id1 in ids1, id2 in ids2
      sf1.hs[s][id1, id2] += sf1.d[id1, s] * sf2.d[id2, s]
      sf1.hs[s][id2, id1] += sf1.d[id1, s] * sf2.d[id2, s]
    end
  end

  # Chain rule for first derivatives
  #fill!(sf1.d, 0.0)
  for s=1:S
    for id1 in ids1
      sf1.d[id1, s] = sf2.v * sf1.d[id1, s]
    end
    for id2 in ids2
      sf1.d[id2, s] += sf1.v * sf2.d[id2, s]
    end
  end

  sf1.v = sf1.v * sf2.v
end
