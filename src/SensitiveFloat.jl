
export multiply_sfs!, add_scaled_sfs!, combine_sfs!, add_sources_sf!

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
  sp::SensitiveFloat{ParamType, NumType}; clear_hessian::Bool=true)
    sp.v = zero(NumType)
    fill!(sp.d, zero(NumType))
    if clear_hessian
      fill!(sp.h, zero(NumType))
    end
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


@doc """
Factor out the hessian part of combine_sfs! to help the compiler.

TODO: I think this is a red herring and this can be put back in
""" ->
function combine_sfs_hessian!{ParamType <: CelesteTypes.ParamSet,
                              T1 <: Number, T2 <: Number, T3 <: Number}(
    sf1::SensitiveFloat{ParamType, T1},
    sf2::SensitiveFloat{ParamType, T1},
    sf_result::SensitiveFloat{ParamType, T1},
    g_d::Vector{T2}, g_h::Matrix{T3})

  p1, p2 = size(sf_result.h)
  @assert size(sf_result.h) == size(sf1.h) == size(sf2.h)
  @assert p1 == p2 == prod(size(sf1.d)) == prod(size(sf2.d))
  for ind2 = 1:p2
    sf11_factor = g_h[1, 1] * sf1.d[ind2] + g_h[1, 2] * sf2.d[ind2]
    sf21_factor = g_h[1, 2] * sf1.d[ind2] + g_h[2, 2] * sf2.d[ind2]

    @inbounds for ind1 = 1:ind2
      sf_result.h[ind1, ind2] =
        g_d[1] * sf1.h[ind1, ind2] +
        g_d[2] * sf2.h[ind1, ind2] +
        sf11_factor * sf1.d[ind1] +
        sf21_factor * sf2.d[ind1]
      sf_result.h[ind2, ind1] = sf_result.h[ind1, ind2]
    end
  end
end


@doc """
Updates sf_result in place with g(sf1, sf2), where
g_d = (g_1, g_2) is the gradient of g and
g_h = (g_11, g_12; g_12, g_22) is the hessian of g,
each evaluated at (sf1, sf2).

The result is stored in sf_result.  The order is done in such a way that
it can overwrite sf1 or sf2 and still be accurate.
""" ->
function combine_sfs!{ParamType <: CelesteTypes.ParamSet,
                      T1 <: Number, T2 <: Number, T3 <: Number}(
    sf1::SensitiveFloat{ParamType, T1},
    sf2::SensitiveFloat{ParamType, T1},
    sf_result::SensitiveFloat{ParamType, T1},
    v::T1, g_d::Vector{T2}, g_h::Matrix{T3};
    calculate_hessian::Bool=true)

  # TODO: time consuming **************

  # TODO: this line is allocating a lot of memory and I don't know why.
  # Commenting this line out attributes the same allocation to the next line.
  # Is memory being allocated lazily or misattributed?
  @assert g_h[1, 2] == g_h[2, 1]

  # You have to do this in the right order to not overwrite needed terms.
  if calculate_hessian
    combine_sfs_hessian!(sf1, sf2, sf_result, g_d, g_h);
  end

  for ind in eachindex(sf_result.d)
    sf_result.d[ind] = g_d[1] * sf1.d[ind] + g_d[2] * sf2.d[ind]
  end

  sf_result.v = v
end


@doc """
Updates sf1 in place with g(sf1, sf2), where
g_d = (g_1, g_2) is the gradient of g and
g_h = (g_11, g_12; g_12, g_22) is the hessian of g,
each evaluated at (sf1, sf2).

The result is stored in sf1.
""" ->
function combine_sfs!{ParamType <: CelesteTypes.ParamSet,
                      T1 <: Number, T2 <: Number, T3 <: Number}(
    sf1::SensitiveFloat{ParamType, T1},
    sf2::SensitiveFloat{ParamType, T1},
    v::T1, g_d::Vector{T2}, g_h::Matrix{T3};
    calculate_hessian::Bool=true)

  combine_sfs!(sf1, sf2, sf1, v, g_d, g_h, calculate_hessian=calculate_hessian)
end

# Decalare outside to avoid allocating memory.
const multiply_sfs_hess = Float64[0 1; 1 0]

@doc """
TODO: don't ignore the ids arguments and test.
""" ->
function multiply_sfs!{ParamType <: CelesteTypes.ParamSet, NumType <: Number}(
    sf1::SensitiveFloat{ParamType, NumType},
    sf2::SensitiveFloat{ParamType, NumType};
    ids1::Vector{Int64}=collect(1:length(ParamType)),
    ids2::Vector{Int64}=collect(1:length(ParamType)),
    calculate_hessian::Bool=true)

  v = sf1.v * sf2.v
  g_d = NumType[sf2.v, sf1.v]
  #const g_h = NumType[0 1; 1 0]

  combine_sfs!(sf1, sf2, v, g_d, multiply_sfs_hess, calculate_hessian=calculate_hessian)
end


@doc """
Update sf1 in place with (sf1 + scale * sf2).
""" ->
function add_scaled_sfs!{ParamType <: CelesteTypes.ParamSet, NumType <: Number}(
    sf1::SensitiveFloat{ParamType, NumType},
    sf2::SensitiveFloat{ParamType, NumType};
    scale::Float64=1.0, calculate_hessian::Bool=true)

  sf1.v = sf1.v + scale * sf2.v

  @inbounds for i in eachindex(sf1.d)
    sf1.d[i] = sf1.d[i] + scale * sf2.d[i]
  end

  if calculate_hessian
    p1, p2 = size(sf1.h)
    @assert (p1, p2) == size(sf2.h)
    @inbounds for ind2=1:p2, ind1=1:ind2
      sf1.h[ind1, ind2] += scale * sf2.h[ind1, ind2]
      sf1.h[ind2, ind1] = sf1.h[ind1, ind2]
    end
  end
end


@doc """
Adds sf2_s to sf1, where sf1 is sensitive to multiple sources and sf2_s is only
sensitive to source s.
""" ->
function add_sources_sf!{ParamType <: CelesteTypes.ParamSet, NumType <: Number}(
    sf_all::SensitiveFloat{ParamType, NumType},
    sf_s::SensitiveFloat{ParamType, NumType},
    s::Int64; calculate_hessian::Bool=true)

  # TODO: This line, too, allocates a lot of memory.  Why?
  sf_all.v = sf_all.v + sf_s.v

  # TODO: time consuming **************
  P = length(ParamType)
  Ph = size(sf_all.h)[1]

  @assert Ph == prod(size(sf_all.d))
  @assert Ph == size(sf_all.h)[2]
  @assert Ph >= P * s
  @assert size(sf_s.d) == (P, 1)
  @assert size(sf_s.h) == (P, P)

  @inbounds for s_ind1 in 1:P
    s_all_ind1 = P * (s - 1) + s_ind1
    sf_all.d[s_all_ind1] = sf_all.d[s_all_ind1] + sf_s.d[s_ind1]
    if calculate_hessian
      @inbounds for s_ind2 in 1:s_ind1
        s_all_ind2 = P * (s - 1) + s_ind2
        sf_all.h[s_all_ind2, s_all_ind1] =
          sf_all.h[s_all_ind2, s_all_ind1] + sf_s.h[s_ind2, s_ind1]
        sf_all.h[s_all_ind1, s_all_ind2] = sf_all.h[s_all_ind2, s_all_ind1]
      end
    end
  end
end
