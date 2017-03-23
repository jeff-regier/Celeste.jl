module SensitiveFloats

export SensitiveFloat,
       SingleSourceSensitiveFloat,
       clear!,
       multiply_sfs!,
       add_scaled_sfs!,
       combine_sfs!,
       add_sources_sf!,
       set_hess!,
       SSparseSensitiveFloat,
       AbstractSensitiveFloat

using Celeste: ParameterizedArray, @aliasscope, Const
using StaticArrays

abstract type AbstractSensitiveFloat{NumType} end
zeros_type(::Type{Array{T,N}} where N, dims...) where T = zeros(T, dims...)
zeros_type(T::Type{ParameterizedArray{x,A}} where x, dims...) where A = T(zeros_type(A, dims...))
zeros_type(aT::Type{SizedArray{S,T,N,M}} where {S,N}, dims...) where {T,M} = aT(zeros_type(Array{T,M}, dims...))

abstract type AbstractSparseSSensitiveFloat{NumType} <: AbstractSensitiveFloat{NumType} end

immutable Source
    n::Int
end

# Special case for local_S == 1
immutable SingleSourceSensitiveFloat{NumType, ParamSet, HessianRepresentation} <: AbstractSparseSSensitiveFloat{NumType}
   v::Base.RefValue{NumType}

   # local_S vector of local_P gradients
   d::ParameterizedArray{ParamSet, Vector{NumType}}

   # local_S x local_S matrix of hessians (generally local_P x local_P matrices)
   h::HessianRepresentation

   has_gradient::Bool
   has_hessian::Bool
end
function (::Type{SingleSourceSensitiveFloat{NumType, ParamSet, HessianRepresentation}}){NumType, ParamSet, HessianRepresentation}(has_gradient, has_hessian)
    @assert has_gradient || !has_hessian
    local_P = length(ParamSet)
    v = Ref(zero(NumType))
    d = zeros_type(ParameterizedArray{ParamSet, Vector{NumType}}, local_P * has_gradient)
    h_dim = local_P
    h = zeros_type(HessianRepresentation, h_dim, h_dim)
    SingleSourceSensitiveFloat{NumType, ParamSet, HessianRepresentation}(v, d, h, has_gradient, has_hessian)
end
n_sources(sf::SingleSourceSensitiveFloat) = 1
n_local_params(sf::SingleSourceSensitiveFloat{NumType, ParamSet} where NumType) where {ParamSet} = isa(ParamSet, Int) ? ParamSet : length(ParamSet)
Base.getindex(sf::SingleSourceSensitiveFloat, s::Source) = (@assert s.n == 1; sf)

# multiple SSSF that are sparse in S
immutable SSparseSensitiveFloat{NumType, ParamSet, HessianRepresentation} <: AbstractSparseSSensitiveFloat{NumType}
    v::Base.RefValue{NumType}

    # local_S vector of local_P gradients
    d::Vector{ParameterizedArray{ParamSet, Vector{NumType}}}

    # local_S x local_S matrix of hessians (generally local_P x local_P matrices)
    h::Vector{HessianRepresentation}

    local_S::Int

    has_gradient::Bool
    has_hessian::Bool
    function (::Type{SSparseSensitiveFloat{NumType, ParamSet, HessianRepresentation}}){NumType, ParamSet, HessianRepresentation}(local_S, has_gradient, has_hessian)
        @assert has_gradient || !has_hessian
        local_P = length(ParamSet)
        v = Ref(zero(NumType))
        d = [zeros_type(ParameterizedArray{ParamSet, Vector{NumType}}, local_P * has_gradient) for i = 1:local_S]
        h_dim = local_P
        h = [zeros_type(HessianRepresentation, h_dim, h_dim) for i = 1:local_S]
        new{NumType, ParamSet, HessianRepresentation}(v, d, h, local_S, has_gradient, has_hessian)
    end
end
n_sources(sf::SSparseSensitiveFloat) = sf.local_S
n_local_params(sf::SSparseSensitiveFloat{NumType, ParamSet} where NumType) where {ParamSet} = isa(ParamSet, Int) ? ParamSet : length(ParamSet)
# This shares the underlying data, so modifications will show up
Base.getindex(sf::SSparseSensitiveFloat{NumType, ParamSet, HessianRepresentation}, s::Source) where {NumType, ParamSet, HessianRepresentation} = 
    SingleSourceSensitiveFloat{NumType, ParamSet, HessianRepresentation}(
      sf.v,
      sf.d[s.n],
      sf.h[s.n],
      sf.has_gradient, sf.has_hessian
    )

immutable SourceViewArray{ParamSet, AT}
    s::Int
    a::AT
end
@inline Base.getindex(a::SourceViewArray{ParamSet}, inds...) where ParamSet = getindex(a.a,
  map(x->x+length(ParamSet)*(a.s-1),Base.to_indices(a.a, inds))...)
@inline Base.setindex!(a::SourceViewArray{ParamSet}, v, inds...) where ParamSet = setindex!(a.a,
  v, map(x->x+length(ParamSet)*(a.s-1),Base.to_indices(a.a, inds))...)

immutable SourceViewSensitiveFloat{NumType, ParamSet}
    v::Base.RefValue{NumType}    
    d::SourceViewArray{ParamSet, Matrix{NumType}}
    h::SourceViewArray{ParamSet, Matrix{NumType}}
end

"""
A function value and its derivative with respect to its arguments.

Attributes:
  v:  The value
  d:  The derivative with respect to each variable in
      P-dimensional VariationalParams for each of S celestial objects
      in a local_P x local_S matrix.
  h:  The second derivative with respect to each variational parameter,
      in the same format as d.  This is used for the full Hessian
      with respect to all the sources.
"""
immutable SensitiveFloat{NumType, ParamSet} <: AbstractSensitiveFloat{NumType}
    v::Base.RefValue{NumType}

    # local_P x local_S matrix of gradients
    d::Matrix{NumType}

    # h is ordered so that p changes fastest.  For example, the indices
    # of a column of h correspond to the indices of d's stacked columns.
    h::Matrix{NumType}

    local_S::Int64

    has_gradient::Bool
    has_hessian::Bool

    function (::Type{SensitiveFloat{NumType, ParamSet}}){NumType, ParamSet}(local_S, has_gradient, has_hessian)
        @assert has_gradient || !has_hessian
        v = Ref(zero(NumType))
        local_P = isa(ParamSet, Integer) ? ParamSet : length(ParamSet)
        d = zeros(NumType, local_P * has_gradient, local_S * has_gradient)
        h_dim = local_P * local_S * has_hessian
        h = zeros(NumType, h_dim, h_dim)
        new{NumType, ParamSet}(v, d, h, local_S, has_gradient, has_hessian)
    end
end
n_sources(sf::SensitiveFloat) = sf.local_S
n_local_params(sf::SensitiveFloat{NumType, ParamSet} where NumType) where {ParamSet} = isa(ParamSet, Int) ? ParamSet : length(ParamSet)
@inline Base.getindex(sf::SensitiveFloat{NumType, ParamSet}, s::Source) where {NumType, ParamSet} =
  SourceViewSensitiveFloat{NumType, ParamSet}(sf.v,
    SourceViewArray{ParamSet, typeof(sf.d)}(s.n, sf.d),
    SourceViewArray{ParamSet, typeof(sf.h)}(s.n, sf.h))

function SensitiveFloat(local_S::Int64,
                        has_gradient::Bool = true,
                        has_hessian::Bool = true)
    return SensitiveFloat{Float64}(local_S, has_gradient, has_hessian)
end

function SensitiveFloat{NumType <: Number, ParamSet}(prototype_sf::SensitiveFloat{NumType, ParamSet})
    SensitiveFloat{NumType, ParamSet}(prototype_sf.local_S,
                                      prototype_sf.has_gradient,
                                      prototype_sf.has_hessian)
end

SensitiveFloat(prototype_sf::SingleSourceSensitiveFloat) = 
  typeof(prototype_sf)(prototype_sf.has_gradient, prototype_sf.has_hessian)

  SensitiveFloat(prototype_sf::SSparseSensitiveFloat) = 
    typeof(prototype_sf)(prototype_sf.local_S, prototype_sf.has_gradient, prototype_sf.has_hessian)

#########################################################

"""
Set a SensitiveFloat's hessian term, maintaining symmetry.
"""
function set_hess!{NumType <: Number}(
                    sf::SensitiveFloat{NumType},
                    i::Int,
                    j::Int,
                    v::NumType)
    @assert sf.has_hessian
    # even if i == j, it's probably faster not to branch
    sf.h[i, j] = sf.h[j, i] = v
end


function zero!{T}(m::Union{Array{T},SizedArray{S,T,N,M} where {S,N,M}})
    for i in eachindex(m)
        @inbounds m[i] = zero(T)
    end
end
zero!(a::ParameterizedArray) = zero!(a.arr)

function clear!{NumType <: Number}(sf::SensitiveFloat{NumType})
    sf.v[] = zero(NumType)

    if sf.has_gradient
        zero!(sf.d)
    end

    if sf.has_hessian
        zero!(sf.h)
    end
end

function clear!{NumType <: Number}(sf::SingleSourceSensitiveFloat{NumType, ParamSet, HessianRepresentation} where {ParamSet, HessianRepresentation})
    sf.v[] = zero(NumType)

    if sf.has_gradient
        zero!(sf.d)
    end

    if sf.has_hessian
        zero!(sf.h)
    end
end

function clear!{NumType <: Number}(sf::SSparseSensitiveFloat{NumType, ParamSet, HessianRepresentation} where {ParamSet, HessianRepresentation})
    sf.v[] = zero(NumType)

    if sf.has_gradient
        for arr in sf.d
            zero!(arr)
        end
    end

    if sf.has_hessian
        for arr in sf.h
            zero!(arr)
        end
    end
end


"""
Factor out the hessian part of combine_sfs!.
"""
function combine_sfs_hessian!{T1 <: Number, T2 <: Number, T3 <: Number}(
            sf1::AbstractSparseSSensitiveFloat{T1},
            sf2::AbstractSparseSSensitiveFloat{T1},
            sf_result::AbstractSensitiveFloat{T1},
            g_d::Vector{T2},
            g_h::Matrix{T3})
    @assert g_h[1, 2] == g_h[2, 1]
    @assert sf_result.has_hessian
    @assert sf_result.has_gradient

    @assert n_sources(sf1) == n_sources(sf2) == n_sources(sf_result)    
    P = n_local_params(sf_result)
    @inbounds for source_i in 1:n_sources(sf_result)
          for ind2 in 1:n_local_params(sf_result)
              sf11_factor = g_h[1, 1] * sf1[Source(source_i)].d[ind2] + g_h[1, 2] * sf2[Source(source_i)].d[ind2]
              sf21_factor = g_h[1, 2] * sf1[Source(source_i)].d[ind2] + g_h[2, 2] * sf2[Source(source_i)].d[ind2]
              for source_j in 1:n_sources(sf_result)
                sf1_s = sf1[Source(source_j)]
                sf2_s = sf2[Source(source_j)]
                if source_i == source_j
                  for ind1 = 1:n_local_params(sf_result)
                      var =
                        sf11_factor * sf1_s.d[ind1] + sf21_factor * sf2_s.d[ind1]
                      var +=
                            g_d[1] * sf1_s.h[ind1, ind2] +
                            g_d[2] * sf2_s.h[ind1, ind2]
                      sf_result.h[(source_j-1) * P + ind1, (source_i-1) * P + ind2] = var
                  end                  
                else
                  for ind1 = 1:n_local_params(sf_result)
                      var = sf11_factor * sf1_s.d[ind1] + sf21_factor * sf2_s.d[ind1]
                      sf_result.h[(source_j-1) * P + ind1, (source_i-1) * P + ind2] = var
                  end
                end
            end 
        end
    end
end

function combine_sfs_gradient!(sf1::AbstractSensitiveFloat{T1},
            sf2::AbstractSensitiveFloat{T1},
            sf_result::AbstractSensitiveFloat{T1},
            g_d) where T1
    for ind in eachindex(sf_result.d)
        sf_result.d[ind] = g_d[1] * sf1.d[ind] + g_d[2] * sf2.d[ind]
    end
end


function combine_sfs_gradient!(sf1::AbstractSparseSSensitiveFloat{T1},
            sf2::AbstractSparseSSensitiveFloat{T1},
            sf_result::AbstractSensitiveFloat{T1},
            g_d) where T1
    P = n_local_params(sf_result)
    for source in 1:n_sources(sf_result)
        for ind in 1:n_local_params(sf_result)
            sf_result.d[(source - 1)*P + ind] =
                g_d[1] * sf1[Source(source)].d[ind] + g_d[2] * sf2[Source(source)].d[ind]
        end
    end
end

"""
Updates sf_result in place with g(sf1, sf2), where
g_d = (g_1, g_2) is the gradient of g and
g_h = (g_11, g_12; g_12, g_22) is the hessian of g,
each evaluated at (sf1, sf2).

The result is stored in sf_result.  The order is done in such a way that
it can overwrite sf1 or sf2 and still be accurate.
"""
function combine_sfs!{T1 <: Number, T2 <: Number, T3 <: Number}(
                        sf1::AbstractSensitiveFloat{T1},
                        sf2::AbstractSensitiveFloat{T1},
                        sf_result::AbstractSensitiveFloat{T1},
                        v::T1,
                        g_d::Vector{T2},
                        g_h::Matrix{T3})
    # You have to do this in the right order to not overwrite needed terms.
    if sf_result.has_hessian
        combine_sfs_hessian!(sf1, sf2, sf_result, g_d, g_h)
    end

    if sf_result.has_gradient
        combine_sfs_gradient!(sf1, sf2, sf_result, g_d)
#=
        sf_result.d[:] = sf1.d
        n = length(sf_result.d)
        @show (n, g_d[1], sf_result.d, 1)
        LinAlg.BLAS.scal!(n, g_d[1], sf_result.d, 1)
        LinAlg.BLAS.axpy!(g_d[2], sf2.d, sf_result.d)
=#
    end

    sf_result.v[] = v
end


# Decalare outside to avoid allocating memory.
const multiply_sfs_hess = Float64[0 1; 1 0]

"""
Updates sf1 in place with sf1 * sf2.
"""
function multiply_sfs!{NumType <: Number}(sf1::AbstractSensitiveFloat{NumType},
                                          sf2::AbstractSensitiveFloat{NumType})
    v = sf1.v[] * sf2.v[]
    g_d = NumType[sf2.v[], sf1.v[]]
    combine_sfs!(sf1, sf2, sf1, v, g_d, multiply_sfs_hess)
end


"""
Update sf1 in place with (sf1 + scale * sf2).
"""
function add_scaled_sfs!{NumType <: Number}(
                    sf1::SensitiveFloat{NumType},
                    sf2::SensitiveFloat{NumType},
                    scale::AbstractFloat)
    sf1.v[] += scale * sf2.v[]

    @assert sf1.has_gradient == sf2.has_gradient
    @assert sf1.has_hessian == sf2.has_hessian

    if sf1.has_gradient
        LinAlg.BLAS.axpy!(scale, sf2.d, sf1.d)
    end

    if sf1.has_hessian
        p1, p2 = size(sf1.h)
        @assert (p1, p2) == size(sf2.h)
        @inbounds for ind2=1:p2, ind1=1:ind2
            sf1.h[ind1, ind2] += scale * sf2.h[ind1, ind2]
            sf1.h[ind2, ind1] = sf1.h[ind1, ind2]
        end
    end

    true # Set definite return type
end

function add_scaled_sfs!{NumType <: Number}(
                    sf1::SensitiveFloat{NumType},
                    sf2::SSparseSensitiveFloat{NumType},
                    scale::AbstractFloat)
    sf1.v[] += scale * sf2.v[]

    @assert sf1.has_gradient == sf2.has_gradient
    @assert sf1.has_hessian == sf2.has_hessian

    P = n_local_params(sf1)
    @inbounds if sf1.has_gradient
        for source in 1:n_sources(sf1)
            for ind in 1:n_local_params(sf1)
                sf1.d[(source-1)*P+ind] += scale * sf2[Source(source)].d[ind]
            end
        end
    end

    if sf1.has_hessian
      @aliasscope begin
        @inbounds for source in 1:n_sources(sf1)
            sf2_s_h = Const(sf2[Source(source)].h.arr)
            for ind1 in 1:n_local_params(sf1)
                for ind2 in 1:n_local_params(sf1)
                    @fastmath sf1.h[(source-1)*P+ind2, (source-1)*P+ind1] += scale * sf2_s_h[ind2, ind1]
                end
            end
        end
      end
    end

    true # Set definite return type
end

"""
Adds sf2_s to sf1, where sf1 is sensitive to multiple sources and sf2_s is only
sensitive to source s.
"""
function add_sources_sf!{NumType <: Number, ParamSet}(
                    sf_all::SensitiveFloat{NumType, ParamSet},
                    sf_s::SensitiveFloat{NumType},
                    s::Int)
    sf_all.v[] += sf_s.v[]

    @assert size(sf_all.d, 1) == size(sf_s.d, 1)

    P = length(ParamSet)
    P_shifted = P * (s - 1)

    if sf_all.has_gradient
        @assert size(sf_s.d) == (P, 1)
        @inbounds for s_ind1 in 1:P
            s_all_ind1 = P_shifted + s_ind1
            sf_all.d[s_all_ind1] = sf_all.d[s_all_ind1] + sf_s.d[s_ind1]
        end
    end

    if sf_all.has_hessian
        Ph = size(sf_all.h)[1]
        @assert Ph == size(sf_all.h)[2]
        @assert size(sf_s.h) == (P, P)
        @assert Ph >= P * s

        @inbounds for s_ind1 in 1:P
            s_all_ind1 = P_shifted + s_ind1

            @inbounds for s_ind2 in 1:P
                s_all_ind2 = P_shifted + s_ind2
                sf_all.h[s_all_ind2, s_all_ind1] += sf_s.h[s_ind2, s_ind1]
                # TODO: move outside the loop?
                # sf_all.h[s_all_ind1, s_all_ind2] = sf_all.h[s_all_ind2, s_all_ind1]
            end
        end
    end
end

function zero_sensitive_float_array(NumType::DataType, ::Type{ParamSet},
                                    local_S::Int,
                                    d::Integer...) where ParamSet
    sf_array = Array{SensitiveFloat{NumType, ParamSet}}(d)
    for ind in 1:length(sf_array)
        # Do we always want these arrays to have gradients and hessians?
        sf_array[ind] = SensitiveFloat{NumType, ParamSet}(local_S, true, true)
    end
    sf_array
end

end
