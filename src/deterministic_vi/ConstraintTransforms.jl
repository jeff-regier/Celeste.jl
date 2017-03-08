module ConstraintTransforms

using ..Model
using ..SensitiveFloats
using ..DeterministicVI: VariationalParams
using Compat

####################
# Constraint Types #
####################

@compat abstract type Constraint end

immutable BoxConstraint <: Constraint
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

immutable SimplexConstraint <: Constraint
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

immutable ParameterConstraint{C<:Constraint} <: Constraint
    constraint::C
    indices::UnitRange{Int} # indices in the bound parameterization
end

ParameterConstraint(c::Constraint, i::Vector) = ParameterConstraint(c, first(i):last(i))
ParameterConstraint(c::Constraint, i::Int) = ParameterConstraint(c, i:i)

immutable ConstraintBatch
    boxes::Vector{Vector{ParameterConstraint{BoxConstraint}}}
    simplexes::Vector{Vector{ParameterConstraint{SimplexConstraint}}}
    function ConstraintBatch(boxes, simplexes)
        @assert length(boxes) == length(simplexes)
        return new(boxes, simplexes)
    end
end

# default ConstraintBatch for Celeste's model
function ConstraintBatch{T}(bound::VariationalParams{T},
                            loc_width::Real = 1.0e-4,
                            loc_scale::Real = 1.0)
    n_sources = length(bound)
    boxes = Vector{Vector{ParameterConstraint{BoxConstraint}}}(n_sources)
    simplexes = Vector{Vector{ParameterConstraint{SimplexConstraint}}}(n_sources)
    for src in 1:n_sources
        u1, u2 = u_ParameterConstraints(bound[src], loc_width, loc_scale)
        boxes[src] = [
            u1,
            u2,
            ParameterConstraint(BoxConstraint(1e-2, 0.99, 1.0), ids.e_dev),
            ParameterConstraint(BoxConstraint(1e-2, 0.99, 1.0), ids.e_axis),
            ParameterConstraint(BoxConstraint(-10.0, 10.0, 1.0), ids.e_angle),
            ParameterConstraint(BoxConstraint(0.10, 70.0, 1.0), ids.e_scale),
            ParameterConstraint(BoxConstraint(-1.0, 10.0, 1.0), ids.r1),
            ParameterConstraint(BoxConstraint(1e-4, 0.10, 1.0), ids.r2),
            ParameterConstraint(BoxConstraint(-10.0, 10.0, 1.0), ids.c1[:, 1]),
            ParameterConstraint(BoxConstraint(-10.0, 10.0, 1.0), ids.c1[:, 2]),
            ParameterConstraint(BoxConstraint(1e-4, 1.0, 1.0), ids.c2[:, 1]),
            ParameterConstraint(BoxConstraint(1e-4, 1.0, 1.0), ids.c2[:, 2])
        ]
        simplexes[src] = [
            ParameterConstraint(SimplexConstraint(0.005, 1.0, 2), ids.a),
            ParameterConstraint(SimplexConstraint(0.01/D, 1.0, D), ids.k[:, 1]),
            ParameterConstraint(SimplexConstraint(0.01/D, 1.0, D), ids.k[:, 2])
        ]
    end
    return ConstraintBatch(boxes, simplexes)
end

function u_BoxConstraint(loc, loc_width, loc_scale)
    return BoxConstraint(loc - loc_width, loc + loc_width, loc_scale)
end

function u_ParameterConstraints(params, loc_width, loc_scale)
    i1, i2 = ids.u[1], ids.u[2]
    u1 = ParameterConstraint(u_BoxConstraint(params[i1], loc_width, loc_scale), i1)
    u2 = ParameterConstraint(u_BoxConstraint(params[i2], loc_width, loc_scale), i2)
    return u1, u2
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

# to_bound!

function to_bound!{T<:Real}(bound::Vector{T}, bound_range::UnitRange,
                            free::Vector{T}, free_range::UnitRange,
                            c::SimplexConstraint)
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

# to_free!

function to_free!{T<:Real}(free::Vector{T}, free_range::UnitRange,
                           bound::Vector{T}, bound_range::UnitRange,
                           c::SimplexConstraint)
    log_last = log(unsimplexify(bound[bound_range[end]], c))
    for i in 1:length(free_range)
        ith_term = log(unsimplexify(bound[bound_range[i]], c)) - log_last
        free[free_range[i]] = c.scale * ith_term
    end
    return free
end

# ParameterConstraint #
#---------------------#

# to_bound!

function to_bound!{T<:Real}(bound::Vector{T}, free::Vector{T},
                            c::ParameterConstraint{BoxConstraint},
                            free_index::Integer)
    for i in c.indices
        bound[i] = to_bound(free[free_index], c.constraint)
        free_index += 1
    end
    return free_index
end

function to_bound!{T<:Real}(bound::Vector{T}, free::Vector{T},
                            c::ParameterConstraint{SimplexConstraint},
                            free_index::Integer)
    free_length = (c.constraint.n - 1)
    free_range = (1:free_length) + free_index - 1
    to_bound!(bound, c.indices, free, free_range, c.constraint)
    return free_index + free_length
end

# to_free!

function to_free!{T<:Real}(free::Vector{T}, bound::Vector{T},
                           c::ParameterConstraint{BoxConstraint},
                           free_index::Integer)
    for i in c.indices
        free[free_index] = to_free(bound[i], c.constraint)
        free_index += 1
    end
    return free_index
end

function to_free!{T<:Real}(free::Vector{T}, bound::Vector{T},
                           c::ParameterConstraint{SimplexConstraint},
                           free_index::Integer)
    free_length = (c.constraint.n - 1)
    free_range = (1:free_length) + free_index - 1
    to_free!(free, free_range, bound, c.indices, c.constraint)
    return free_index + free_length
end


# ConstraintBatch #
#-----------------#

# to_bound!

function to_bound!{T<:Real}(bound::Vector{T}, free::Vector{T},
                            constraints::ConstraintBatch, src::Integer)
    free_index = 1
    for constraint in constraints.boxes[src]
        free_index = to_bound!(bound, free, constraint, free_index)
    end
    for constraint in constraints.simplexes[src]
        free_index = to_bound!(bound, free, constraint, free_index)
    end
end

function to_bound!{T}(bound::VariationalParams{T}, free::VariationalParams{T},
                      constraints::ConstraintBatch)
    @assert length(free) == length(bound)
    for src in 1:length(free)
        to_bound!(bound[src], free[src], constraints, src)
    end
end

# to_free!

function to_free!{T<:Real}(free::Vector{T}, bound::Vector{T},
                           constraints::ConstraintBatch, src::Integer)
    free_index = 1
    for constraint in constraints.boxes[src]
        free_index = to_free!(free, bound, constraint, free_index)
    end
    for constraint in constraints.simplexes[src]
        free_index = to_free!(free, bound, constraint, free_index)
    end
end

function to_free!{T}(free::VariationalParams{T}, bound::VariationalParams{T},
                     constraints::ConstraintBatch)
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

function enforce!{T<:Real}(bound::Vector{T}, c::ParameterConstraint{BoxConstraint})
    for i in c.indices
        bound[i] = enforce(bound[i], c.constraint)
    end
    return bound
end

function enforce!{T<:Real}(bound::Vector{T}, c::ParameterConstraint{SimplexConstraint})
    enforce!(view(bound, c.indices), c.constraint)
    return bound
end

# ConstraintBatch #
#-----------------#

function enforce!{T<:Real}(bound::Vector{T}, constraints::ConstraintBatch, src::Integer)
    for constraint in constraints.boxes[src]
        enforce!(bound, constraint)
    end
    for constraint in constraints.simplexes[src]
        enforce!(bound, constraint)
    end
end

function enforce!{T}(bound::VariationalParams{T}, constraints::ConstraintBatch)
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

function allocate_free_params{T}(bound::VariationalParams{T}, constraints::ConstraintBatch)
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

immutable TransformJacobianBundle{N,T}
    jacobian::Matrix{T}
    output::Vector{T}
    cfg::JacobianConfig{N,T,Tuple{Vector{Dual{N,T}},Vector{Dual{N,T}}}}
end

# this is a good chunk size because it divides evenly into `length(CanonicalParams)`
const DEFAULT_CHUNK = 11

if length(CanonicalParams) % DEFAULT_CHUNK != 0
    warn("""
         It looks like length(CanonicalParams) changed; you'll want to update DEFAULT_CHUNK
         in ConstraintTransforms.jl (currently $DEFAULT_CHUNK) to a reasonable chunk size
         for AD purposes (ping @jrevels for help).
         """)
end

function TransformJacobianBundle{T<:Real,N}(output::Vector{T}, input::Vector{T},
                                            ::Type{Val{N}} = Val{DEFAULT_CHUNK})
    jacobian = zeros(T, length(output), length(input))
    cfg = JacobianConfig{N}(output, input)
    return TransformJacobianBundle(jacobian, similar(output), cfg)
end

function TransformJacobianBundle{T<:Real,N}(output::VariationalParams{T},
                                         input::VariationalParams{T},
                                         ::Type{Val{N}} = Val{DEFAULT_CHUNK})
    @assert length(output) == length(input)
    @assert all(length(src) == length(output[1]) for src in output)
    @assert all(length(src) == length(input[1]) for src in input)
    return TransformJacobianBundle(output[1], input[1], Val{N})
end

# Propagate derivatives of `transform!` back from the output-parameterized SensitiveFloat
# (`sf_output`) to the input-parameterized SensitiveFloat (`sf_input`). Note that the memory
# for the raw output parameters is supplied via `cfg`, whose constructor is
# `TransformJacobianBundle(output_params, input_params)`.
function propagate_derivatives!{F,N,T}(transform!::F,
                                       sf_output::SensitiveFloat{T},
                                       sf_input::SensitiveFloat{T},
                                       input_sources::VariationalParams{T},
                                       constraints::ConstraintBatch,
                                       bundle::TransformJacobianBundle{N,T})
    sf_input.v[] = sf_output.v[]
    n_sources = length(input_sources)
    n_input_params = size(sf_input.d, 1)
    n_output_params = size(sf_output.d, 1)
    # Julia v0.5 erroneously fails to type-infer `src` for some reason, so we type-annotate
    # it explicitly here. It still unnecessarily boxes/unboxes `src` for some of the uses
    # in the body of the loop, but I'm not sure what to do for that.
    for src::Int64 in 1:n_sources
        jacobian!(bundle.jacobian,
                  (y, x) -> transform!(y, x, constraints, src),
                  bundle.output,
                  input_sources[src],
                  bundle.cfg)
        backprop_jacobian!(sf_input.d, sf_output.d, bundle.jacobian, src)
    end
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

end # module
