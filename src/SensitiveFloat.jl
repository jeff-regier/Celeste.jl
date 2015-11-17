
type HessianEntry{NumType <: Number}
  # An entry in a sparse Hessian containing a derivative with respect
  # to source s1, parameter i1 and source s2, parameter i2,
  # with value v.
  s1::Int64
  i1::Int64
  s2::Int64
  i2::Int64
  v::NumType
end


@doc """
A function value and its derivative with respect to its arguments.

Attributes:
  v:  The value
  d:  The derivative with respect to each variable in
      P-dimensional VariationalParams for each of S celestial objects
      in a local_P x local_S matrix.
  h:  The second derivative with respect to each variational parameter,
      in the same format as d.  This is used for the full Hessian
      with respect to all the sources.
  hs: An array of per-source Hessians.  This will generally be reserved
      for the Hessian of brightness values that depend only on one source.
""" ->
type SensitiveFloat{ParamType <: ParamSet, NumType <: Number}
    v::NumType
    d::Matrix{NumType} # local_P x local_S
    h::Array{HessianEntry{NumType}}
    hs::Array{Matrix{NumType}} #  local_S array of local_P x local_P
    ids::ParamType
end

#########################################################

@doc """
Set a SensitiveFloat's hessian term, maintaining symmetry.
""" ->
function set_hess!{ParamType <: ParamSet, NumType <: Number}(
    sf::SensitiveFloat{ParamType, NumType},
    i::Int64, j::Int64, v::NumType)
  set_hess!(sf, i, j, 1, v)
end

function set_hess!{ParamType <: ParamSet, NumType <: Number}(
    sf::SensitiveFloat{ParamType, NumType},
    i::Int64, j::Int64, s::Int64, v::NumType)
  i != j ?
    sf.hs[s][i, j] = sf.hs[s][j, i] = v:
    sf.hs[s][i, j] = v
end

function zero_sensitive_float{ParamType <: ParamSet}(
  ::Type{ParamType}, NumType::DataType, local_S::Int64)
    local_P = length(ParamType)
    d = zeros(NumType, local_P, local_S)
    h = HessianEntry{NumType}[]

    # TODO: is there some kind of symmetric matrix type to use?
    hs = fill(zeros(local_P, local_P), local_S)
    SensitiveFloat{ParamType, NumType}(
      zero(NumType), d, h, hs, getids(ParamType))
end

function zero_sensitive_float{ParamType <: ParamSet}(
  ::Type{ParamType}, NumType::DataType)
    # Default to a single source.
    zero_sensitive_float(ParamType, NumType, 1)
end

function clear!{ParamType <: ParamSet, NumType <: Number}(
  sp::SensitiveFloat{ParamType, NumType})
    sp.v = zero(NumType)
    h = HessianEntry{NumType}[]
    fill!(sp.d, zero(NumType))
    [ fill!(sp.hs[s], zero(NumType)) for s=1:size(sp.d)[2] ]
end

# If no type is specified, default to using Float64.
function zero_sensitive_float{ParamType <: ParamSet}(
  param_arg::Type{ParamType}, local_S::Int64)
    zero_sensitive_float(param_arg, Float64, local_S)
end

function zero_sensitive_float{ParamType <: ParamSet}(param_arg::Type{ParamType})
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
function multiply_sf!{NumType <: Number}(
    sf1::SensitiveFloat{NumType}, sf2::SensitiveFloat{NumType};
    ids1::Array{Int64}=collect(1:length(sf1.ids)),
    ids2::Array{Int64}=collect(1:length(sf2.ids)))

  S = size(ids1.d)[2]
  sf1.v = sf1.v * sf2.v

  # Chain rule for first derivatives
  fill!(sf1.d, 0.0)
  for s=1:S
    for id1 in ids1
      sf1.d[id1, s] += sf2.v * sf1.d[id1, s]
    end
    for id2 in ids2
      sf1.d[id2, s] += sf1.v * sf2.d[id2, s]
    end
  end

  # Chain rule for second derivatives.
  for s=1:S
    fill!(sf1.hs[s], 0.0)

    # Second derivative terms involving first derivatives.
    for id1 in ids1, id2 in ids2
      sf1.hs[s][id2, id1] = sf1.hs[s][id1, id2] += sf1.d[id1, s] * sf2.d[id2, s]
    end

    # Second derivate terms involving second derivates.  TODO: nearly half are redudnant.
    for i in ids1, j in ids1
      sf1.hs[i, j] += sf2.v * sf1.hs[s][i, j]
    end
    for i in ids2, j in ids2
      sf1.hs[i, j] += sf1.v * sf2.hs[s][i, j]
    end
  end
end
j
