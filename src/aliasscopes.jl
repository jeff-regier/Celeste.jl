using StaticArrays

if isdefined(Base.SimdLoop, Symbol("@unroll_annotation"))
struct Const{T}
    a::T
end

Base.getindex(A::Const{<:Array}, i1::Int) = Core.const_arrayref(A.is_star, i1)
@inline Base.getindex(A::Const{<:Array}, i1::Int, i2::Int, I::Int...) =  Core.const_arrayref(A.is_star, i1, i2, I...)

@generated function Base.getindex{SM<:SizedMatrix}(m::Const{SM}, i1::Integer, i2::Integer)
    quote
      $(Expr(:meta, :inline, :propagate_inbounds))
      data = m.is_star.data
      @boundscheck if (i1 < 1 || i1 > $(size(SM,1)) || i2 < 1 || i2 > $(size(SM, 2)))
          throw(BoundsError(data, (i1,i2)))
      end

      @inbounds return Core.const_arrayref(data, i1 + $(size(SM,1))*(i2-1))
    end
end

function compile_unroll(x)
    (isa(x, Expr) && x.head == :for) || throw(SimdError("for loop expected"))
    length(x.args) == 2 || throw(SimdError("1D for loop expected"))
    var,range = Base.SimdLoop.parse_iteration_space(x.args[1])
    r = gensym("r")
    quote
        let $r = $range
            local $var = first($r)
            while $var <= last($r)
                $(x.args[2])        # Body of loop
                $var += 1
                $(Expr(:unrollloop))  # Mark loop as SIMD loop
            end
        end
    end
end

macro unroll_loop(forloop)
  esc(compile_unroll(forloop))
end

macro aliasscope(body)
    sym = gensym()
    esc(quote
        $(Expr(:aliasscope))
        $sym = $body
        $(Expr(:popaliasscope))
        $sym
    end)
end
else
# Compatibility definitions when compiler enhancements not present
Const(x) = x
macro unroll_loop(x)
  esc(x)
end
macro aliasscope(body)
  esc(body)
end
end
