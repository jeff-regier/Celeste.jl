module ConstraintTransforms

using ..Model
using ..SensitiveFloats
using ..DeterministicVI: VariationalParams
using Celeste: Const, @aliasscope
using Compat

####################
# Constraint Types #
####################

abstract type Constraint end

struct BoxConstraint <: Constraint
    lower::Float64
    upper::Float64
    scale::Float64
    function BoxConstraint(lower, upper, scale)
        @assert lower > -Inf
        @assert lower < upper
        @assert upper < Inf
        @assert 0.0 < scale < Inf
        return new(lower, upper, scale)
    end
end

struct SimplexConstraint <: Constraint
    lower::Float64
    scale::Float64
    n::Int # length of the simplex in the free parameterization
    function SimplexConstraint(lower, scale, n)
        @assert n >= 2
        @assert 0.0 <= lower < 1/n
        @assert 0.0 < scale < Inf
        return new(lower, scale, n)
    end
end

struct ParameterConstraint{C<:Constraint} <: Constraint
    constraint::C
    indices::UnitRange{Int} # indices in the bound parameterization
end

ParameterConstraint(c::Constraint, i::Vector) = ParameterConstraint(c, first(i):last(i))
ParameterConstraint(c::Constraint, i::Int) = ParameterConstraint(c, i:i)

struct ConstraintBatch
    boxes::Vector{Vector{ParameterConstraint{BoxConstraint}}}
    simplexes::Vector{Vector{ParameterConstraint{SimplexConstraint}}}
    function ConstraintBatch(boxes, simplexes)
        @assert length(boxes) == length(simplexes)
        return new(boxes, simplexes)
    end
end

#################################################
# `to_bound`/`to_bound!` & `to_free`/`to_free!` #
#################################################

# BoxConstraint #
#---------------#

# to_bound

logit(x) = 1.0 / (1.0 + exp(-x))

function to_bound(free::Real, c::BoxConstraint)
    @assert isfinite(free)
    return logit(free / c.scale) * (c.upper - c.lower) + c.lower
end

# to_free

inv_logit(x) = -log(1.0 / x - 1)

function to_free(bound::Real, c::BoxConstraint)
    @assert c.lower < bound < c.upper
    return inv_logit((bound - c.lower) / (c.upper - c.lower)) * c.scale
end

# SimplexConstraint #
#-------------------#

simplexify(x::Real, c::SimplexConstraint) = (1 - c.n * c.lower) * x + c.lower

unsimplexify(x::Real, c::SimplexConstraint) = (x - c.lower) / (1 - c.n * c.lower)


function to_bound!(bound::Vector{T}, bound_range::UnitRange,
                   free::Vector{T}, free_range::UnitRange,
                   c::SimplexConstraint) where {T<:Real}
    for i in 1:length(free_range)
        x = free[free_range[i]]
        @assert isfinite(x)
        bound[bound_range[i]] = x / c.scale
    end
    m = bound[bound_range[1]]
    for i in 1:length(free_range)
        m = max(bound[bound_range[i]], m)
    end
    exp_neg_m = exp(-m)
    bound_sum = exp_neg_m
    for i in bound_range[1:end-1]
        x = exp(bound[i] - m)
        bound[i] = x
        bound_sum += x
    end
    for i in bound_range[1:end-1]
        bound[i] = simplexify(bound[i] / bound_sum, c)
    end
    bound[bound_range[end]] = simplexify(inv(bound_sum) * exp_neg_m, c)
    return bound
end


function to_free!(free::Vector{T}, free_range::UnitRange,
                  bound::Vector{T}, bound_range::UnitRange,
                  c::SimplexConstraint) where {T<:Real}
    log_last = log(unsimplexify(bound[bound_range[end]], c))
    for i in 1:length(free_range)
        ith_term = log(unsimplexify(bound[bound_range[i]], c)) - log_last
        free[free_range[i]] = c.scale * ith_term
    end
    return free
end

# ParameterConstraint #
#---------------------#


function to_bound!(bound::Vector{T}, free::Vector{T},
                   c::ParameterConstraint{BoxConstraint},
                   free_index::Integer) where {T<:Real}
    for i in c.indices
        bound[i] = to_bound(free[free_index], c.constraint)
        free_index += 1
    end
    return free_index
end


function to_bound!(bound::Vector{T}, free::Vector{T},
                   c::ParameterConstraint{SimplexConstraint},
                   free_index::Integer) where {T<:Real}
    free_length = (c.constraint.n - 1)
    free_range = (1:free_length) + free_index - 1
    to_bound!(bound, c.indices, free, free_range, c.constraint)
    return free_index + free_length
end


function to_free!(free::Vector{T}, bound::Vector{T},
                  c::ParameterConstraint{BoxConstraint},
                  free_index::Integer) where {T<:Real}
    for i in c.indices
        free[free_index] = to_free(bound[i], c.constraint)
        free_index += 1
    end
    return free_index
end


function to_free!(free::Vector{T}, bound::Vector{T},
                  c::ParameterConstraint{SimplexConstraint},
                  free_index::Integer) where {T<:Real}
    free_length = (c.constraint.n - 1)
    free_range = (1:free_length) + free_index - 1
    to_free!(free, free_range, bound, c.indices, c.constraint)
    return free_index + free_length
end


# ConstraintBatch #
#-----------------#

function to_bound!(bound::Vector{T}, free::Vector{T},
                   constraints::ConstraintBatch, src::Integer) where {T<:Real}
    free_index = 1
    for constraint in constraints.boxes[src]
        free_index = to_bound!(bound, free, constraint, free_index)
    end
    for constraint in constraints.simplexes[src]
        free_index = to_bound!(bound, free, constraint, free_index)
    end
end


function to_bound!(bound::VariationalParams{T}, free::VariationalParams{T},
                   constraints::ConstraintBatch) where {T}
    @assert length(free) == length(bound)
    for src in 1:length(free)
        to_bound!(bound[src], free[src], constraints, src)
    end
end


function to_free!(free::Vector{T}, bound::Vector{T},
                  constraints::ConstraintBatch, src::Integer) where {T<:Real}
    free_index = 1
    for constraint in constraints.boxes[src]
        free_index = to_free!(free, bound, constraint, free_index)
    end
    for constraint in constraints.simplexes[src]
        free_index = to_free!(free, bound, constraint, free_index)
    end
end


function to_free!(free::VariationalParams{T}, bound::VariationalParams{T},
                  constraints::ConstraintBatch) where {T}
    @assert length(free) == length(bound)
    for src in 1:length(free)
        to_free!(free[src], bound[src], constraints, src)
    end
end

########################
# `enforce`/`enforce!` #
########################

# BoxConstraint #
#---------------#

function enforce(bound::Number, c::BoxConstraint)
    if !(c.lower < bound < c.upper)
        # don't set the value to exact bounds to avoid Infs during transforms
        return max(min(bound, prevfloat(c.upper)), nextfloat(c.lower))
    end
    return bound
end

# SimplexConstraint #
#-------------------#

function enforce!(bound, c::SimplexConstraint)
    bound_sum = zero(eltype(bound))
    for i in eachindex(bound)
        if !(c.lower < bound[i] < 1.0)
            # don't set the value to exact bounds to avoid Infs during transforms
            bound[i] = max(min(bound[i], prevfloat(1.0)), nextfloat(c.lower))
        end
        bound_sum += bound[i]
    end
    if !(isapprox(bound_sum, one(eltype(bound))))
        # normalize in a way that maintains the lower bounds
        rescale = (1 - c.n * c.lower) / (bound_sum - c.n * c.lower)
        for i in eachindex(bound)
            bound[i] = nextfloat(c.lower) + rescale * (bound[i] - c.lower)
        end
    end
    return bound
end

# ParameterConstraint #
#---------------------#

function enforce!(bound::Vector{T}, c::ParameterConstraint{BoxConstraint}) where {T<:Real}
    for i in c.indices
        bound[i] = enforce(bound[i], c.constraint)
    end
    return bound
end


function enforce!(bound::Vector{T}, c::ParameterConstraint{SimplexConstraint}) where {T<:Real}
    enforce!(view(bound, c.indices), c.constraint)
    return bound
end


# ConstraintBatch #
#-----------------#

function enforce!(bound::Vector{T},
                  constraints::ConstraintBatch,
                  src::Integer) where {T<:Real}
    for constraint in constraints.boxes[src]
        enforce!(bound, constraint)
    end
    for constraint in constraints.simplexes[src]
        enforce!(bound, constraint)
    end
end

function enforce!(bound::VariationalParams{T},
                  constraints::ConstraintBatch) where {T}
    for src in 1:length(bound)
        enforce!(bound[src], constraints, src)
    end
end

##########################
# `allocate_free_params` #
##########################

function free_length(boxes::Vector{ParameterConstraint{BoxConstraint}})
    return sum(length(box.indices) for box in boxes)
end

function free_length(simplexes::Vector{ParameterConstraint{SimplexConstraint}})
    return sum(simplex.constraint.n - 1 for simplex in simplexes)
end

function allocate_free_params(bound::VariationalParams{T}, constraints::ConstraintBatch) where {T}
    free = similar(bound)
    for src in 1:length(bound)
        n_simplex_params = free_length(constraints.simplexes[src])
        n_box_params = free_length(constraints.boxes[src])
        free[src] = zeros(T, n_simplex_params + n_box_params)
    end
    return free
end

###################
# Differentiation #
###################

using ForwardDiff
using ForwardDiff: Dual, jacobian!, JacobianConfig

const NUM_ELBO_BOUND_PARAMS = length(CanonicalParams)

const TransformJacobianConfig{M,N,T} = JacobianConfig{N,T,Tuple{Array{Dual{N,T},M},Vector{Dual{N,T}}}}

struct TransformDerivatives{N,T}
    jacobian::Matrix{T}
    hessian::Array{T,3}
    output_dual::Vector{Dual{N,T}}
    inner_cfg::TransformJacobianConfig{1,N,Dual{N,T}}
    outer_cfg::TransformJacobianConfig{2,N,T}
end

# this is a good chunk size because it divides evenly into `length(CanonicalParams)`
const DEFAULT_CHUNK = 11

function TransformDerivatives(
    output::Vector{T}, input::Vector{T},
    ::Type{Val{N}} = Val{DEFAULT_CHUNK}) where {T<:Real,N}

    jacobian = zeros(T, length(output), length(input))
    hessian = zeros(T, length(output), length(input), length(input))
    output_dual = copy!(similar(output, Dual{N,T}), output)
    inner_cfg = JacobianConfig{N}(output_dual, similar(input, Dual{N,T}))
    outer_cfg = JacobianConfig{N}(jacobian, input)
    return TransformDerivatives{N,T}(jacobian, hessian, output_dual, inner_cfg, outer_cfg)
end

function TransformDerivatives(
    output::VariationalParams{T},
    input::VariationalParams{T},
    ::Type{Val{N}} = Val{DEFAULT_CHUNK}) where {T<:Real,N}

    @assert length(output) == length(input)
    @assert all(length(src) == length(output[1]) for src in output)
    @assert all(length(src) == length(input[1]) for src in input)
    return TransformDerivatives(output[1], input[1], Val{N})
end

function differentiate!(transform!::F,
                        derivs::TransformDerivatives,
                        input::Vector{T}) where {F,T<:Number}
    jacobian!(reshape(derivs.hessian, length(derivs.jacobian), length(input)),
              (out, x) -> jacobian!(out, transform!, derivs.output_dual, x, derivs.inner_cfg),
              derivs.jacobian, input, derivs.outer_cfg)
    return derivs
end

# Propagate derivatives of `transform!` back from the output-parameterized SensitiveFloat
# (`sf_output`) to the input-parameterized SensitiveFloat (`sf_input`). Note that the memory
# for the raw output parameters is supplied via `derivs`, whose constructor is
# `TransformDerivatives(output_params, input_params)`.
function propagate_derivatives!(transform!::F,
                                sf_output::SensitiveFloat,
                                sf_input::SensitiveFloat,
                                input_sources::VariationalParams{T},
                                constraints::ConstraintBatch,
                                derivs::TransformDerivatives) where {F,T}
    sf_input.v[] = sf_output.v[]
    n_sources = length(input_sources)
    n_input_params = size(sf_input.d, 1)
    n_output_params = size(sf_output.d, 1)
    input_hessian = reshape(sf_input.h, n_sources, n_input_params, n_sources, n_input_params)
    output_hessian = reshape(sf_output.h, n_sources, n_output_params, n_sources, n_output_params)
    # Julia v0.5 erroneously fails to type-infer `src` for some reason, so we type-annotate
    # it explicitly here. It still unnecessarily boxes/unboxes `src` for some of the uses
    # in the body of the loop, but I'm not sure what to do for that.
    for src::Int64 in 1:n_sources
        differentiate!((y, x) -> transform!(y, x, constraints, src), derivs, input_sources[src])
        backprop_jacobian!(sf_input.d, sf_output.d, derivs.jacobian, src)
        backprop_hessian!(input_hessian, output_hessian, derivs.hessian, derivs.jacobian, sf_output.d, src)
    end
    symmetrize!(sf_input.h)
    return sf_input
end

# equivalent to At_mul_B!(input[:, src], jacobian, output[:, src])
function backprop_jacobian!(input, output, jacobian, src)
    for i in 1:size(input, 1)
        x = zero(eltype(input))
        for j in 1:size(output, 1)
            @inbounds x += jacobian[j, i] * output[j, src]
        end
        @inbounds input[i, src] = x
    end
    return input
end

function backprop_hessian!(input, output, hessian, jacobian, gradient, src)
    first_quad_form!(input, jacobian, output, src)
    for j in 1:size(hessian, 2)
        for k in 1:size(hessian, 3)
            x = zero(eltype(input))
            for i in 1:size(hessian, 1)
                @inbounds x += hessian[i, j, k] * gradient[i, src]
            end
            @inbounds input[src, j, src, k] += x
        end
    end
    return input
end

# equivalent to `At_mul_B!(view(C, src, :, src, :), A, view(B, src, :, src, :) * A)`
# this could be further optimized by assuming the symmetry of `view(B, src, :, src, :)`
function first_quad_form!(C, A, B, src)
    const m = NUM_ELBO_BOUND_PARAMS
    @assert m == size(A, 1)
    n = size(A, 2)
    scratch = Array{Float64, 2}(m, m)
    @aliasscope begin
        for i in 1:m, j in 1:m
            scratch[i, j] = Const(B)[src, j, src, i]
        end
    end
    @aliasscope begin
        for i in 1:n, j in 1:n
            x = zero(eltype(C))
            for k in 1:m
                y = zero(eltype(C))
                @inbounds for l in 1:m
                    @fastmath y += Const(scratch)[l, k] * Const(A)[l, j]
                end
                @inbounds x += Const(A)[k, i] * y
            end
            @inbounds C[src, i, src, j] = x
        end
    end
    return C
end

# ensure exact symmetry (necessary for some numerical linear algebra routines)
function symmetrize!(A, c = 0.5)
    for i in 1:size(A, 1), j in 1:i
        @inbounds A[i, j] = A[j, i] = c * (A[i, j] + A[j, i])
    end
    return A
end

end # module
