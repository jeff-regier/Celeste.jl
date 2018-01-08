module SensitiveFloats

export SensitiveFloat,
       multiply_sfs!,
       add_scaled_sfs!,
       combine_sfs!,
       add_sources_sf!,
       set_hess!


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
struct SensitiveFloat{T}
    v::Base.RefValue{T}

    # local_P x local_S matrix of gradients
    d::Matrix{T}

    # h is ordered so that p changes fastest.  For example, the indices
    # of a column of h correspond to the indices of d's stacked columns.
    h::Matrix{T}

    local_P::Int64
    local_S::Int64

    has_gradient::Bool
    has_hessian::Bool

    function SensitiveFloat{T}(local_P, local_S, has_gradient, has_hessian) where {T}
        @assert has_gradient || !has_hessian
        v = Ref(zero(T))
        d = zeros(T, local_P * has_gradient, local_S * has_gradient)
        h_dim = local_P * local_S * has_hessian
        h = zeros(T, h_dim, h_dim)
        new(v, d, h, local_P, local_S, has_gradient, has_hessian)
    end
end

function SensitiveFloat(local_P::Int64, local_S::Int64,
                        has_gradient::Bool = true,
                        has_hessian::Bool = true)
    return SensitiveFloat{Float64}(local_P, local_S, has_gradient, has_hessian)
end

function SensitiveFloat(prototype_sf::SensitiveFloat{T}) where {T}
    SensitiveFloat{T}(prototype_sf.local_P,
                      prototype_sf.local_S,
                      prototype_sf.has_gradient,
                      prototype_sf.has_hessian)
end

#########################################################

"""
Set a SensitiveFloat's hessian term, maintaining symmetry.
"""
function set_hess!(sf::SensitiveFloat{T},
                   i::Int,
                   j::Int,
                   v::T) where {T<:Number}
    @assert sf.has_hessian
    # even if i == j, it's probably faster not to branch
    sf.h[i, j] = sf.h[j, i] = v
end


function zero!(m::Array{T}) where {T}
    for i in eachindex(m)
        @inbounds m[i] = zero(T)
    end
end

function zero!(sf::SensitiveFloat{T}) where {T<:Number}
    sf.v[] = zero(T)

    if sf.has_gradient
        zero!(sf.d)
    end

    if sf.has_hessian
        zero!(sf.h)
    end
end


"""
Factor out the hessian part of combine_sfs!.
"""
function combine_sfs_hessian!(
        sf1::SensitiveFloat{T1},
        sf2::SensitiveFloat{T1},
        sf_result::SensitiveFloat{T1},
        g_d::Vector{T2},
        g_h::Matrix{T3}) where {T1<:Number,T2<:Number,T3<:Number}
    p1, p2 = size(sf_result.h)

    @assert size(sf_result.d) == size(sf1.d) == size(sf2.d)
    @assert size(sf_result.h) == size(sf1.h) == size(sf2.h)
    @assert p1 > 0
    @assert g_h[1, 2] == g_h[2, 1]
    @assert sf_result.has_hessian
    @assert sf_result.has_gradient

    for ind2 = 1:p2
        sf11_factor = g_h[1, 1] * sf1.d[ind2] + g_h[1, 2] * sf2.d[ind2]
        sf21_factor = g_h[1, 2] * sf1.d[ind2] + g_h[2, 2] * sf2.d[ind2]

        @inbounds for ind1 = 1:p2
            sf_result.h[ind1, ind2] =
                g_d[1] * sf1.h[ind1, ind2] +
                g_d[2] * sf2.h[ind1, ind2] +
                sf11_factor * sf1.d[ind1] +
                sf21_factor * sf2.d[ind1]
            #sf_result.h[ind2, ind1] = sf_result.h[ind1, ind2]
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
function combine_sfs!(
        sf1::SensitiveFloat{T1},
        sf2::SensitiveFloat{T1},
        sf_result::SensitiveFloat{T1},
        v::T1,
        g_d::Vector{T2},
        g_h::Matrix{T3}) where {T1<:Number,T2<:Number,T3<:Number}
    # You have to do this in the right order to not overwrite needed terms.
    if sf_result.has_hessian
        combine_sfs_hessian!(sf1, sf2, sf_result, g_d, g_h)
    end

    if sf_result.has_gradient
        for ind in eachindex(sf_result.d)
            sf_result.d[ind] = g_d[1] * sf1.d[ind] + g_d[2] * sf2.d[ind]
        end
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
function multiply_sfs!(sf1::SensitiveFloat{T},
                       sf2::SensitiveFloat{T}) where {T<:Number}
    v = sf1.v[] * sf2.v[]
    g_d = T[sf2.v[], sf1.v[]]
    combine_sfs!(sf1, sf2, sf1, v, g_d, multiply_sfs_hess)
end


"""
Update sf1 in place with (sf1 + scale * sf2).
"""
function add_scaled_sfs!(sf1::SensitiveFloat{T},
                         sf2::SensitiveFloat{T},
                         scale::AbstractFloat) where {T<:Number}
    sf1.v[] += scale * sf2.v[]

    @assert sf1.has_gradient == sf2.has_gradient
    @assert sf1.has_hessian == sf2.has_hessian
    @assert size(sf1.h) == size(sf2.h)

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


"""
Adds sf2_s to sf1, where sf1 is sensitive to multiple sources and sf2_s is only
sensitive to source s.
"""
function add_sources_sf!(sf_all::SensitiveFloat{T},
                         sf_s::SensitiveFloat{T},
                         s::Int) where {T<:Number}
    sf_all.v[] += sf_s.v[]

    @assert size(sf_all.d, 1) == size(sf_s.d, 1)

    P = sf_all.local_P
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


function zero_sensitive_float_array(::Type{T},
                                    local_P::Int,
                                    local_S::Int,
                                    d::Integer...) where {T}
    sf_array = Array{SensitiveFloat{T}}(d)
    for ind in 1:length(sf_array)
        # Do we always want these arrays to have gradients and hessians?
        sf_array[ind] = SensitiveFloat{T}(local_P, local_S, true, true)
    end
    sf_array
end

end
