struct Param{Set, name, dims}; end
@Base.pure Base.length(x::Param{Set, name, dims} where {Set, name}) where {dims} = reduce(*, 1, dims)
@Base.pure Base.length(x::Type{Param{Set, name, dims}} where {Set, name}) where {dims} = reduce(*, 1, dims)
to_batch(p::Param) = p

struct ParamBatch{T<:Tuple}
    params::T
end
to_batch(p::ParamBatch) = p
@Base.pure Base.length(x::ParamBatch) = mapreduce(length, +, 0, typeof(x).parameters[1].parameters)
@Base.pure Base.length(T::Type{<:ParamBatch}) = mapreduce(length, +, 0, T.parameters[1].parameters)
@Base.pure function Base.getindex(x::ParamBatch, y::Param)
    idx = 1
    for (i,p) in enumerate(x.params)
      p == y && return length(p) == 1 ? idx : SVector{length(p),Int}(tuple((idx:(idx+length(p)-1))...))
      idx += length(p)
    end
    error("Not found")
end
@Base.pure function Base.getindex(x::ParamBatch, y::ParamBatch)
    reduce(static_vcat, SVector{0,Int}(()), (x[p] for p in y.params))
end
@Base.pure Base.getindex(x::ParamBatch, y) = Base.getindex(x, to_batch(y))

struct ParameterizedArray{ParamSet, A}
    arr::A
end
(::Type{ParameterizedArray{ParamSet,X} where X})(a::A) where {ParamSet, A} = 
  ParameterizedArray{ParamSet, A}(a)
Base.size(a::ParameterizedArray) = Base.size(a.arr)
Base.eachindex(a::ParameterizedArray) = Base.eachindex(a.arr)
Base.length(a::ParameterizedArray) = Base.length(a.arr)
@inline Base.@propagate_inbounds Base.getindex(a::Const{<:ParameterizedArray}, inds...) = Const(a.a.arr)[inds...]
@inline Base.@propagate_inbounds Base.getindex(a::ParameterizedArray, inds...) = a.arr[to_indices(a, inds)...]
@inline Base.@propagate_inbounds Base.setindex!(a::ParameterizedArray, v, inds...) = Base.setindex!(a.arr,
  v, to_indices(a, inds)...)
@Base.pure function get_axis_parameterization(tA, tInds)
    params = tA.parameters[1]
    params.parameters[Base.sub_int(Base.add_int(1, params.parameters.length), tInds.parameters.length)]
end
@inline function Base.to_indices(A::ParameterizedArray{<:Tuple}, inds, I::Tuple{Any, Vararg{Any}})
      axis_parametrization = get_axis_parameterization(typeof(A), typeof(inds))
      (Base.to_index(axis_parametrization, I[1]),
        Base.to_indices(A, Base._maybetail(inds), Base.tail(I))...)
end
Base.transpose(a::ParameterizedArray{ParamSet}) where {ParamSet} = ParameterizedArray{ParamSet}(transpose(a.arr))
import Base: *, -, +
for op in (:(*),:(-),:(+))
  @eval begin
    $(op)(a::ParameterizedArray, b) = $(op)(a.arr, b)
    $(op)(a, b::ParameterizedArray) = $(op)(a, b.arr)
    $(op)(a::ParameterizedArray, b::ParameterizedArray) = $(op)(a.arr, b.arr)
  end
end
Base.size(arr::ParameterizedArray, dim) = Base.size(arr.arr, dim)
Base.issymmetric(arr::ParameterizedArray) = Base.issymmetric(arr.arr)

function normalize_param(xT, T)
    if T.parameters[1] <: Tuple && xT == T.parameters[1].parameters[1]
        Param{Base.unwrap_unionall(T.parameters[1].parameters[1]).name.wrapper,
          T.parameters[2], T.parameters[3]}
    else
        T
    end
end

function process_params!(xT, T, cur_start_idx, params, oT = T)
    if !(nfields(T) > 0)
      error("Type $oT had non-parameter fields")
    end
    for (fName, fT) in map(i->(fieldname(T, i), fieldtype(T, i)), 1:nfields(T))
        if !(fT <: Param)
            cur_start_idx = process_params!(xT, fT, cur_start_idx, params, oT)
            continue
        end
        dims = fT.parameters[3]
        nparams = reduce(*, 1, dims)
        params[normalize_param(xT, fT)] = length(dims) == 0 ? cur_start_idx :
          SVector{nparams}(collect(cur_start_idx:(cur_start_idx + nparams -1)))
        cur_start_idx += nparams
    end
    cur_start_idx
end

function to_batch
end

struct SubParam{T}
    inds::Tuple
end
Base.getindex(x::T, inds...) where {T<:Param} = SubParam{T}(inds)
@Base.pure @inline Base.to_index(a, x::SubParam) = Base.to_index(a, typeof(x).parameters[1]())[x.inds...]

macro concretize(xT)
    T = Base.unwrap_unionall(xT)
    @assert isa(T, DataType)
    tname = gensym(T.name.name)
    cur_start_idx = 1
    params = ObjectIdDict()
    cur_start_idx = process_params!(xT, T, cur_start_idx, params)
    esc(quote
        (X::Type{x})() where x <: $xT = X($((:(fieldtype(X, $i)()) for i = 1:nfields(T))...))
        let params = $params
          @Base.pure function Base.to_index(::Type{<:$xT}, I::Param)
            x = ($normalize_param)($xT, typeof(I))
            params[x]
          end
        end
        @Base.pure @inline Base.to_index(::ParameterizedArray{<:$xT, <:Any}, I::Param) = Base.to_index($xT, I)
        Base.length(::Type{<:$xT}) = $(cur_start_idx - 1)
        @Base.pure @inline to_batch(x::$xT) = ParamBatch(tuple($((:(getfield(x, $i)) for i = 1:nfields(T))...)))
        @Base.pure @inline Base.to_index(a, I::$xT) = Base.to_index(a, to_batch(I))
    end)
end

function get_fields(T)
    return collect(map(1:nfields(T)) do i
        :($(fieldname(T, i))::$(fieldtype(T, i)))
    end)
end

function with_inline!(def, types)
    for (i, field) in enumerate(def.args[3].args)
        isexpr(field, :line) && continue
        @assert isexpr(field, :(::))
        length(field.args) == 2 && continue
        def.args[3].args[i] = Expr(:block, get_fields(shift!(types))...)
    end
    def
end

function collect_parameters(def)
    params = Any[]
    for (i, field) in enumerate(def.args[3].args)
        isexpr(field, :line) && continue
        @assert isexpr(field, :(::))
        length(field.args) == 2 && continue
        push!(params, strip_quote(field.args[1]))
    end
    params
end

function make_type(def, types...)
    with_inline!(def, collect(types))
end

macro inline_unnamed(def)
    !isa(def, Expr) && return def
    if !isexpr(def, :type)
        for (i, ex) in enumerate(def.args)
            def.args[i] = inline_unnamed(ex)
        end
        return def
    end
    x = :(eval(current_module(), $(make_type)($(Expr(:quote, def)), $(collect_parameters(def)...))))
    esc(x)
end

is_implicitly_symmetric(x) = false

using Base.Meta
function transpose_implicitly(ex::Expr)
    if isexpr(ex, :block)
        Expr(:block, map(transpose_implicitly, ex.args)...)
    elseif (isexpr(ex, :(=)) || isexpr(ex, :(+=))) && isexpr(ex.args[1], :ref)
        lhs = ex.args[1]
        quote
            $ex
            if !is_implicitly_symmetric($(lhs.args[1]))
                $(lhs.args[1])[$(reverse(lhs.args[2:end])...)] = ($lhs)'
            end
        end
    else
        ex
    end
end

macro implicit_transpose(ex)
    esc(transpose_implicitly(ex))
end

@inline function static_vcat(a::Int, b::Int)
    SVector((a,b))
end
@inline static_vcat(a::SVector, b::Int) = vcat(a, SVector((b,)))
@inline static_vcat(a::Int, b::SVector) = vcat(SVector((a,)), b)
@inline static_vcat(a::SVector, b::SVector) = vcat(a, b)

@generated function Base.to_index(A, I::ParamBatch)
    exprs = collect(:(Base.to_index(A,$p())) for p in I.parameters[1].parameters)
    isempty(exprs) && return :(SVector{0,Int}(()))
    expr = exprs[end]
    for x in reverse(exprs[1:end-1])
        expr = :(static_vcat($x, $expr))
    end
    quote
      $(Expr(:meta, :inline))
      $expr
    end
end

function diagonal_block(T, p::Union{Param, ParamBatch})
    nparams = length(typeof(p))
    SArray{Tuple{nparams, nparams}, T, 2, nparams^2}
end
diagonal_block(T, x) = diagonal_block(T, to_batch(x))

function off_diagonal_block(T, p1::Union{Param, ParamBatch},
                               p2::Union{Param, ParamBatch})
    SArray{Tuple{length(p1), length(p2)}, T, 2, length(p1)*length(p2)}
end
off_diagonal_block(T, x1, x2) = off_diagonal_block(T, to_batch(x1), to_batch(x2))

using Base.Meta
macro create_sparse_implementation(T, sT)
    zero_init = map(1:nfields(Base.unwrap_unionall(T))) do i
        :(zero(fieldtype(X, $i)))
    end
    zero_reset = map(1:nfields(Base.unwrap_unionall(T))) do i
        :(setfield!(x, $(quot(fieldname(T, i))), zero(fieldtype(typeof(x), $i))))
    end
    esc(quote
        Base.zeros(X::Type{<:$sT}) = X($(zero_init...))
        zero!(x::$sT) = begin
            $(zero_reset...)
            nothing
        end
    end)
end

function get_actual_def(def)
    if isexpr(def, :block)
        for ex in def.args
            isexpr(ex, :type) && return ex
        end
        return fieldname_matrix
    end
    @assert isexpr(def, :type)
    def
end

function strip_quote(sym)
    if isa(sym, QuoteNode)
        return sym.value
    elseif isexpr(sym, :quote)
        return sym.args[1]
    elseif isa(sym, Symbol)
        return sym
    else
        error()
    end
end

function generate_fieldname_map(param_set, sparse_def)
    sparse_def = get_actual_def(sparse_def)
    fieldname_map = Vector{Tuple{Any,Any,Any}}()
    for field in sparse_def.args[3].args
        isexpr(field, :line) && continue
        @assert isexpr(field, :(::))
        fname = field.args[1]
        typ = field.args[2]
        @assert isexpr(typ, :call)
        if typ.args[1] == :diagonal_block
            col_fieldname = row_fieldname = typ.args[3]
        elseif typ.args[1] == :off_diagonal_block
            row_fieldname = typ.args[3]
            col_fieldname = typ.args[4]
        end
        push!(fieldname_map, (row_fieldname, col_fieldname, fname))
    end
    fieldname_map
end


function generate_indexing_body_for_type(callback, param_set, ids_instance, fieldname_map, allow_transposed)
    ex = ret = nothing
    for (rowName, colName, fname) in fieldname_map
        body = Expr(:if,:(row == $rowName && col == $colName),callback(fname, false))
        if ex == nothing
            ex = ret = body
        else
            push!(ex.args, body)
            ex = body
        end
    end
    if allow_transposed
        for (rowName, colName, fname) in fieldname_map
            body = Expr(:if,:(row == $colName && col == $rowName),callback(fname, true))
            if ex == nothing
                ex = ret = body
            else
                push!(ex.args, body)
                ex = body
            end
        end
    end
    ret
end

fixup(mat::StaticArrays.SArray{dims, T, N, 1} where {dims, T, N}) = mat[]
fixup(mat::StaticArrays.SMatrix{1, 1, T, 1} where T) = mat[]
fixup(mat) = mat

fill_if_scalar(mat::AbstractArray, T) = mat
fill_if_scalar(scalar, T) = fill(scalar, T)
macro define_accessors(param_set, instance_var, def)
    fieldname_map = generate_fieldname_map(param_set, def)
    name = get_actual_def(def).args[2].args[1]
    if isexpr(name, :(<:))
        name = name.args[1]
    end
    setindex_body(field, transposed) = :(return mat.$field = ($fill_if_scalar)(val, typeof(mat.$field)) )
    getindex_body(field, transposed) = transposed ? :(return transpose(mat.$field)) :
                                                    :(return mat.$field)
    a = quote
        $def
        @inline function Base.setindex!(mat::$name, val, row, col)
            #(row, col) = (to_batch(row), to_batch(col))
            ((length(typeof(row)) == 0) || length(typeof(col)) == 0) && return
            $(generate_indexing_body_for_type(setindex_body, param_set, instance_var, fieldname_map, false))
            @show (row, col, length(typeof(col)), length(typeof(row)))
            error("Trying to assign to region that is implicitly zero or below the diagonal")
        end
        @inline function Base.getindex(mat::$name, row, col)
            #(row, col) = (to_batch(row), to_batch(col))
            $(generate_indexing_body_for_type(getindex_body, param_set, instance_var, fieldname_map, true))
            @show (row, col, length(typeof(col)), length(typeof(row)))
            error("Trying to access a region that is implicitly zero or below the diagonal")
        end
    end
    @show a
    esc(a)
end

# Parse iteration space expression
#       symbol '=' range
#       symbol 'in' range
function parse_iteration_space(x)
    (isa(x, Expr) && (x.head == :(=) || x.head == :in)) || throw("= or in expected")
    length(x.args) == 2 || throw("range syntax is wrong")
    x.args # symbol, range
end

function recursive_replace!(sym_f, expr_f, expr::Expr)
    for i = 1:length(expr.args)
        if isa(expr.args[i], Expr)
            expr.args[i] = expr_f(expr.args[i])
            recursive_replace!(sym_f, expr_f, expr.args[i])
            continue
        end
        expr.args[i] = sym_f(expr.args[i])
    end
end

# Hack for now
function lift_range(range)
    if isexpr(range, :tuple)
        range.args
    elseif !isa(range, Expr)
        collect(range)
    else
        error("Failed to lift range $range")
    end
end

macro syntactic_unroll(for_loop)
    isexpr(for_loop, :for) || throw("for loop expected")
    sym, range = parse_iteration_space(for_loop.args[1])
    range = lift_range(range)
    ret = Expr(:block)
    for el in range
        l = gensym()
        next_label = :(@label($l))
        body = deepcopy(for_loop.args[2])
        # AbstractTrees.jl has this function, but let's just leave it here
        replace_continue(expr) = isexpr(expr, :continue) ? :(@goto($l)) : expr
        recursive_replace!(identity, replace_continue, body)
        if isexpr(sym, :tuple)
            expr = body
            for (s,v) in reverse(collect(zip(sym.args, el)))
                expr = quote
                  let $s = $v
                    $expr
                  end
                end
            end
        else
            expr = quote
                let $sym = $el
                    $body
                end
            end
        end
        push!(ret.args, expr)
        push!(ret.args, next_label)
    end
    x = esc(ret)
    x
end
