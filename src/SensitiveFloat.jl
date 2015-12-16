
#using CelesteTypes.getids
#using CelesteTypes.HessianEntry

export multiply_sf!, add_scaled_sfs!, combine_sfs!

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
type SensitiveFloat{ParamType <: CelesteTypes.ParamSet, NumType <: Number}
    v::NumType
    d::Matrix{NumType} # local_P x local_S
    # h is ordered so that p changes fastest.  For example, the indices
    # of a column of h correspond to the indices of d's stacked columns.
    h::Matrix{NumType} # (local_P * local_S) x (local_P * local_S)
    ids::ParamType
end


#########################################################

@doc """
Set a SensitiveFloat's hessian term, maintaining symmetry.
""" ->
function set_hess!{ParamType <: CelesteTypes.ParamSet, NumType <: Number}(
    sf::SensitiveFloat{ParamType, NumType},
    i::Int64, j::Int64, v::NumType)
  i != j ?
    sf.h[i, j] = sf.h[j, i] = v:
    sf.h[i, j] = v
end

function zero_sensitive_float{ParamType <: CelesteTypes.ParamSet}(
  ::Type{ParamType}, NumType::DataType, local_S::Int64)
    local_P = length(ParamType)
    d = zeros(NumType, local_P, local_S)
    h = zeros(NumType, local_P * local_S, local_P * local_S)
    SensitiveFloat{ParamType, NumType}(
      zero(NumType), d, h, getids(ParamType))
end

function zero_sensitive_float{ParamType <: CelesteTypes.ParamSet}(
  ::Type{ParamType}, NumType::DataType)
    # Default to a single source.
    zero_sensitive_float(ParamType, NumType, 1)
end

function clear!{ParamType <: CelesteTypes.ParamSet, NumType <: Number}(
  sp::SensitiveFloat{ParamType, NumType})
    sp.v = zero(NumType)
    fill!(sp.d, zero(NumType))
    fill!(sp.h, zero(NumType))
end

# If no type is specified, default to using Float64.
function zero_sensitive_float{ParamType <: CelesteTypes.ParamSet}(
  param_arg::Type{ParamType}, local_S::Int64)
    zero_sensitive_float(param_arg, Float64, local_S)
end

function zero_sensitive_float{ParamType <: CelesteTypes.ParamSet}(
  param_arg::Type{ParamType})
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
  sf3.h = sf1.h + sf2.h

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
  @assert S == 1 # For now this is only for the brightness calculations.
  # TODO: replace this with combine_sfs!.

  # You have to do this in the right order to not overwrite needed terms.

  # Second derivate terms involving second derivates.
  sf1.h[:, :] = sf1.v * sf2.h + sf2.v * sf1.h
  sf1.h[:, :] += sf1.d[:] * sf2.d[:]' + sf2.d[:] * sf1.d[:]'

  sf1.d[:, :] = sf2.v * sf1.d + sf1.v * sf2.d

  sf1.v = sf1.v * sf2.v
end



@doc """
Updates sf1 in place with g(sf1, sf2), where
g_d = (g_1, g_2) is the gradient of g and
g_h = (g_11, g_12; g_12, g_22) is the hessian of g,
each evaluated at (sf1, sf2).

The result is stored in sf1.
""" ->
function combine_sfs!{ParamType <: CelesteTypes.ParamSet, NumType <: Number}(
    sf1::SensitiveFloat{ParamType, NumType},
    sf2::SensitiveFloat{ParamType, NumType},
    v::NumType, g_d::Vector{NumType}, g_h::Matrix{NumType})

  S = size(sf1.d)[2]
  @assert g_h[1, 2] == g_h[2, 1]

  # You have to do this in the right order to not overwrite needed terms.

  # Chain rule for second derivatives.
  sf1.h[:, :] = g_d[1] * sf1.h + g_d[2] * sf2.h
  sf1.h[:, :] +=
    g_h[1, 1] * sf1.d[:] * sf1.d[:]' +
    g_h[2, 2] * sf2.d[:] * sf2.d[:]' +
    g_h[1, 2] * (sf1.d[:] * sf2.d[:]' + sf2.d[:] * sf1.d[:]')
  for s=1:S
    sf1.d[:, s] = g_d[1] * sf1.d[:, s] + g_d[2] * sf2.d[:, s]
  end

  sf1.v = v
end


@doc """
TODO: don't ignore the ids arguments and test.
""" ->
function multiply_sfs!{ParamType <: CelesteTypes.ParamSet, NumType <: Number}(
    sf1::SensitiveFloat{ParamType, NumType},
    sf2::SensitiveFloat{ParamType, NumType};
    ids1::Vector{Int64}=collect(1:length(ParamType)),
    ids2::Vector{Int64}=collect(1:length(ParamType)))

  v = sf1.v * sf2.v
  g_d = NumType[sf2.v, sf1.v]
  g_h = NumType[0 1; 1 0]

  combine_sfs!(sf1, sf2, v, g_d, g_h)
end


@doc """
Update sf1 in place with (sf1 + scale * sf2).
""" ->
function add_scaled_sfs!{ParamType <: CelesteTypes.ParamSet, NumType <: Number}(
    sf1::SensitiveFloat{ParamType, NumType},
    sf2::SensitiveFloat{ParamType, NumType};
    scale::Float64=1.0)

  sf1.v = sf1.v + scale * sf2.v
  sf1.d = sf1.d + scale * sf2.d
  sf1.h = sf1.h + scale * sf2.h
end
