struct Symmetric2{T<:Real,S<:AbstractVector{T},uplo} <: AbstractMatrix{T}
    data::S
end
zeros_type(X::Type{Symmetric2{T,S,U}} where {T,U}, dim1, dim2) where {S} = X(zeros_type(S,div(dim1*(dim2+1),2)))
zero!(x::Symmetric2) = zero!(x.data)

function Base.size(A::Symmetric2, i::Integer)
    # if i == 1 || i == 2
        m = length(A.data)
        n = 0
        n1 = 1
        while n*n1 < 2m
            n  = n1
            n1 = n + 1
        end
        return n
    # else
        # error("arraysize: dimension out of range")
    # end
end

@inbounds Base.size(A::Symmetric2) = ntuple(i -> size(A, 1), Val{2})

function Base.getindex(A::Symmetric2{T,S,:L} where {T,S}, i::Integer, j::Integer)
    Base.@_propagate_inbounds_meta
    if j > i
        return A[j, i]
    else
        n = size(A, 1)
        return A.data[i + (j - 1)*n - (j*(j - 1) >> 1)]
    end
end

function Base.setindex!(A::Symmetric2{T,S,:L} where {T,S}, x, i::Integer, j::Integer)
    Base.@_propagate_inbounds_meta
    if j > i
        return setindex!(A, x, j, i)
    else
        n = size(A, 1)
        return setindex!(A.data, x, i + (j - 1)*n - (j*(j - 1) >> 1))
    end
end
