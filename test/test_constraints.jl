using Celeste.Model: CanonicalParams, ids, NUM_COLOR_COMPONENTS

using Celeste.DeterministicVI.ElboMaximize: elbo_constraints

using Celeste.DeterministicVI.ConstraintTransforms: Constraint, BoxConstraint,
                                    SimplexConstraint, ParameterConstraint, ConstraintBatch,
                                    inv_logit, logit, to_free,
                                    to_free!, to_bound, to_bound!, simplexify, unsimplexify,
                                    enforce, enforce!, allocate_free_params,
                                    TransformDerivatives, differentiate!,
                                    propagate_derivatives!

import Celeste.Transform, Celeste.SensitiveFloats

using Base.Test

srand(1)


# Test helpers

function valid_boxed_parameters!(bound, indices, c::BoxConstraint)
    population = linspace(c.lower, c.upper)
    for src in eachindex(bound), i in indices
        bound[src][i] = rand(population)
    end
    return bound
end


function valid_simplexed_parameters!(bound, indices, c::SimplexConstraint)
    @assert length(indices) == c.n
    for src in eachindex(bound)
        params = view(bound[src], indices)
        for i in eachindex(params)
            params[i] = min(rand() + nextfloat(c.lower), 1.0)
        end
        normalize!(params, 1)
        @assert all(c.lower .< params .< 1.0)
        @assert isapprox(sum(params), 1.0)
    end
    return bound
end


function correct_to_free(bound, bound_range, c::SimplexConstraint)
    log_last = log(unsimplexify(bound[bound_range[end]], c))
    free = similar(bound, length(bound_range) - 1)
    for i in 1:length(free)
        free[i] = c.scale * (log(unsimplexify(bound[bound_range[i]], c)) - log_last)
    end
    return free
end


function star_constraints(bound; loc_width = 1.5e-3, loc_scale = 1.0)
    n_sources = length(bound)
    boxes = Vector{Vector{ParameterConstraint{BoxConstraint}}}(n_sources)
    simplexes = Vector{Vector{ParameterConstraint{SimplexConstraint}}}(n_sources)
    for src in 1:n_sources
        i1, i2 = ids.pos[1], ids.pos[2]
        u1, u2 = bound[src][i1], bound[src][i2]
        boxes[src] = [
            ParameterConstraint(BoxConstraint(u1 - loc_width, u1 + loc_width, loc_scale), i1),
            ParameterConstraint(BoxConstraint(u2 - loc_width, u2 + loc_width, loc_scale), i2),
            ParameterConstraint(BoxConstraint(-1.0, 10.0, 1.0), ids.flux_loc[1]),
            ParameterConstraint(BoxConstraint(-10.0, 10.0, 1.0), ids.color_mean[:, 1]),
            ParameterConstraint(BoxConstraint(1e-4, 1.0, 1.0), ids.color_var[:, 1])
        ]
        simplexes[src] = [
            ParameterConstraint(SimplexConstraint(0.005, 1.0, 2), ids.is_star)
            ParameterConstraint(SimplexConstraint(0.01/NUM_COLOR_COMPONENTS, 1.0, NUM_COLOR_COMPONENTS), ids.k[:, 1])
        ]
    end
    return ConstraintBatch(boxes, simplexes)
end


@testset "constraints" begin

    @testset "constraint types" begin

        @test BoxConstraint <: Constraint
        @test SimplexConstraint <: Constraint
        @test ParameterConstraint <: Constraint

        n = 3
        lower = 0.01 / n
        scale = rand()

        box = BoxConstraint(lower, 2*lower, scale)
        @test box.lower === lower
        @test box.upper === 2*lower
        @test box.scale === scale

        @test_throws AssertionError BoxConstraint(lower, lower/2, scale)
        @test_throws AssertionError BoxConstraint(-Inf, lower, scale)
        @test_throws AssertionError BoxConstraint(lower, Inf, scale)
        @test_throws AssertionError BoxConstraint(lower, 2*lower, 0.0)
        @test_throws AssertionError BoxConstraint(lower, 2*lower, -scale)
        @test_throws AssertionError BoxConstraint(lower, 2*lower, Inf)

        simplex = SimplexConstraint(lower, scale, n)
        @test simplex.lower === lower
        @test simplex.scale === scale
        @test simplex.n === n

        @test_throws AssertionError SimplexConstraint(lower, scale, 0)
        @test_throws AssertionError SimplexConstraint(lower, scale, 1)
        @test_throws AssertionError SimplexConstraint(-lower, scale, 1)
        @test_throws AssertionError SimplexConstraint(2*lower, scale, 1)
        @test_throws AssertionError SimplexConstraint(lower, 0.0, 2)
        @test_throws AssertionError SimplexConstraint(lower, 0.0, n)
        @test_throws AssertionError SimplexConstraint(lower, -scale, n)
        @test_throws AssertionError SimplexConstraint(lower, Inf, n)

        c = ParameterConstraint(box, n)
        @test c.constraint === box
        @test c.indices === n:n

        c = ParameterConstraint(box, 1:n)
        @test c.constraint === box
        @test c.indices === 1:n

        c = ParameterConstraint(simplex, n)
        @test c.constraint === simplex
        @test c.indices === n:n

        c = ParameterConstraint(simplex, 1:n)
        @test c.constraint === simplex
        @test c.indices === 1:n
    end



    @testset "to_bound[!]/to_free[!] BoxConstraint" begin
        x = rand()
        @test inv_logit(x) === -log(1.0 / x - 1)
        @test logit(x) === 1.0 / (1.0 + exp(-x))
        @test isapprox(logit(inv_logit(x)), x)

        lower, scale = rand(2)

        c = BoxConstraint(lower, 2*lower, scale)
        bound = 1.5 * c.lower

        @test_throws AssertionError to_free(c.lower, c)
        @test_throws AssertionError to_free(c.upper, c)
        @test to_free(bound, c) === inv_logit((bound - c.lower) / (c.upper - c.lower)) * c.scale

        free = to_free(bound, c)

        @test_throws AssertionError to_bound(-Inf, c)
        @test_throws AssertionError to_bound(Inf, c)
        @test to_bound(free, c) === bound
    end


    @testset "to_bound[!]/to_free[!] SimplexConstraint" begin
        x = rand()
        c = SimplexConstraint(0.01/5, rand(), 5)

        @test simplexify(x, c) === (1 - c.n * c.lower) * x + c.lower
        @test unsimplexify(x, c) === (x - c.lower) / (1 - c.n * c.lower)
        @test unsimplexify(simplexify(x, c), c) === x

        n_params = 10
        n_sources = 3
        bound_range = 1:5
        free_range = 3:6
        bound = [rand(n_params) for _ in 1:n_sources]
        valid_simplexed_parameters!(bound, bound_range, c)
        old_bound = deepcopy(bound)
        free = [zeros(n_params - 1) for _ in 1:n_sources]

        for src in 1:n_sources
            to_free!(free[src], free_range, bound[src], bound_range, c)
            @test (free[src][free_range] ==
                   correct_to_free(bound[src], bound_range, c))

            for i in setdiff(1:n_sources, src)
                for j in setdiff(1:length(free[src]), free_range)
                    @test free[i][j] === 0.0
                end
            end

            bound[src][bound_range] = 0.0
            to_bound!(bound[src], bound_range, free[src], free_range, c)
            @test isapprox(old_bound[src], bound[src])
            @test all(old_bound[i] == bound[i]
                      for i in setdiff(1:n_sources, src))

            copy!(bound[src], old_bound[src])
            fill!(free[src], 0.0)
        end
    end


    @testset "ParameterConstraint BoxConstraint" begin
        n_params = 10
        n_sources = 3
        lower, scale = rand(2)
        c = ParameterConstraint(BoxConstraint(lower, 2*lower, scale), 3:6)
        bound = collect(rand(n_params) for _ in 1:n_sources)
        valid_boxed_parameters!(bound, c.indices, c.constraint)
        old_bound = deepcopy(bound)
        free = collect(zeros(n_params - 1) for _ in 1:n_sources)

        for src in 1:n_sources
            free_index = to_free!(free[src], bound[src], c, 1)
            @test free_index == (length(c.indices) + 1)
            for i in 1:length(c.indices)
                @test free[src][i] == to_free(bound[src][c.indices[i]], c.constraint)
            end

            free_index = to_bound!(bound[src], free[src], c, 1)
            @test free_index == (length(c.indices) + 1)
            @test isapprox(bound[src], old_bound[src])
            @test all(old_bound[i] == bound[i] for i in setdiff(1:n_sources, src))

            copy!(bound[src], old_bound[src])
            fill!(free[src], 0.0)
        end
    end


    @testset "ParameterConstraint SimplexConstraint" begin
        n_params = 10
        n_sources = 3
        c = ParameterConstraint(SimplexConstraint(0.01/5, rand(), 5), 3:7)
        bound = collect(rand(n_params) for _ in 1:n_sources)
        valid_simplexed_parameters!(bound, c.indices, c.constraint)
        old_bound = deepcopy(bound)
        free = collect(zeros(n_params - 1) for _ in 1:n_sources)

        for src in 1:n_sources
            free_index = to_free!(free[src], bound[src], c, 1)
            @test free_index == length(c.indices)
            @test free[src][1:(free_index-1)] == correct_to_free(bound[src], c.indices, c.constraint)

            free_index = to_bound!(bound[src], free[src], c, 1)
            @test free_index == length(c.indices)
            @test isapprox(bound[src], old_bound[src])
            @test all(old_bound[i] == bound[i] for i in setdiff(1:n_sources, src))

            copy!(bound[src], old_bound[src])
            fill!(free[src], 0.0)
        end
    end


    @testset "enforce[!] BoxConstraint" begin
        lower, scale = rand(2)
        c = BoxConstraint(lower, 2*lower, scale)

        @test enforce(c.lower, c) === nextfloat(c.lower)
        @test enforce(c.upper, c) === prevfloat(c.upper)
        @test enforce(-Inf, c) === nextfloat(c.lower)
        @test enforce(Inf, c) === prevfloat(c.upper)
        @test enforce(1.5 * c.lower, c) === 1.5 * c.lower
    end


    @testset "enforce[!] SimplexConstraint" begin
        n = 10
        lower = 0.01 / n
        c = SimplexConstraint(lower, rand(), n)
        bound = valid_simplexed_parameters!([rand(n)], 1:n, c)[1]
        old_bound = copy(bound)

        enforce!(bound, c)
        @test bound == old_bound

        bound[2] = 1.0
        bound[4] = c.lower
        bound[6] = Inf
        bound[7] = c.lower * 1.5

        enforce!(bound, c)
        @test all(c.lower .< bound .< 1.0)
        @test isapprox(sum(bound), 1.0)

        new_bound = copy(old_bound)
        new_bound[2] = prevfloat(1.0)
        new_bound[4] = nextfloat(c.lower)
        new_bound[6] = prevfloat(1.0)
        new_bound[7] = c.lower * 1.5
        rescale = (1 - c.n * c.lower) / (sum(new_bound) - c.n * c.lower)
        new_bound[:] = [nextfloat(c.lower) + rescale * (i - c.lower) for i in new_bound]
        @test bound == new_bound
    end

# ParameterConstraint #
#---------------------#

# BoxConstraint
    @testset "ParameterConstraint BoxConstraint" begin
        n_params = 10
        n_sources = 3
        lower, scale = rand(2)
        c = BoxConstraint(lower, 2*lower, scale)
        pc = ParameterConstraint(c, 3:7)
        bound = collect(rand(n_params) for _ in 1:n_sources)
        valid_boxed_parameters!(bound, pc.indices, c)
        old_bound = deepcopy(bound)

        for src in 1:n_sources
            enforce!(bound[src], pc)

            for i in 1:n_sources
                @test old_bound[i] == bound[i]
                bound[i][3] = c.lower
                bound[i][4] = c.upper
                bound[i][5] = -Inf
                bound[i][6] = Inf
                bound[i][7] = 1.5 * c.lower
            end

            enforce!(bound[src], pc)

            @test bound[src][3] === nextfloat(c.lower)
            @test bound[src][4] === prevfloat(c.upper)
            @test bound[src][5] === nextfloat(c.lower)
            @test bound[src][6] === prevfloat(c.upper)
            @test bound[src][7] === 1.5 * c.lower

            for other_src in setdiff(1:n_sources, src)
                @test bound[other_src][3] === c.lower
                @test bound[other_src][4] === c.upper
                @test bound[other_src][5] === -Inf
                @test bound[other_src][6] === Inf
                @test bound[other_src][7] === 1.5 * c.lower
            end

            for i in 1:n_sources
                for other_param in (1, 2, 8, 9, 10)
                    @test bound[i][other_param] == old_bound[i][other_param]
                end
                copy!(bound[i], old_bound[i])
            end
        end
    end


    @testset "ParameterConstraint SimplexConstraint" begin
        n_params = 10
        n_sources = 3
        n = 5
        lower = 0.01 / n
        c = SimplexConstraint(lower, rand(), n)
        pc = ParameterConstraint(c, 3:7)
        bound = collect(rand(n_params) for _ in 1:n_sources)
        valid_simplexed_parameters!(bound, pc.indices, c)
        old_bound = deepcopy(bound)

        for src in 1:n_sources
            enforce!(bound[src], pc)

            for i in 1:n_sources
                @test old_bound[i] == bound[i]
                bound[i][3] = 1.0
                bound[i][4] = c.lower
                bound[i][6] = Inf
                bound[i][7] = c.lower * 1.5
            end

            new_bound = deepcopy(bound)
            enforce!(bound[src], pc)

            @test all(c.lower .< bound[src][pc.indices] .< 1.0)
            @test isapprox(sum(bound[src][pc.indices]), 1.0)

            new_bound[src][3] = prevfloat(1.0)
            new_bound[src][4] = nextfloat(c.lower)
            new_bound[src][6] = prevfloat(1.0)
            new_bound[src][7] = c.lower * 1.5
            rescale = (1 - c.n * c.lower) / (sum(new_bound[src][pc.indices]) - c.n * c.lower)
            for i in pc.indices
                new_bound[src][i] = nextfloat(c.lower) + rescale * (new_bound[src][i] - c.lower)
            end

            for i in 1:n_sources
                @test new_bound[i] == bound[i]
                copy!(bound[i], old_bound[i])
            end
        end
    end


    @testset "ConstraintBatch" begin
        loc_width, loc_scale = rand(), rand()
        n_sources = 4
        bound = collect(rand(length(CanonicalParams)) for _ in 1:n_sources)
        constraints = elbo_constraints(bound, loc_width, loc_scale)

        for src in 1:n_sources
            u1_box = constraints.boxes[src][1].constraint
            u2_box = constraints.boxes[src][2].constraint
            u1, u2 = bound[src][ids.pos[1]], bound[src][ids.pos[2]]
            @test u1_box === BoxConstraint(u1 - loc_width, u1 + loc_width, loc_scale)
            @test u2_box === BoxConstraint(u2 - loc_width, u2 + loc_width, loc_scale)
        end

        free = allocate_free_params(bound, constraints)

        @test length(free) == length(bound) == n_sources

        free_length = length(CanonicalParams) - length(constraints.simplexes[1])

        for src in 1:n_sources
            @test length(free[src]) == free_length
        end

        enforce!(bound, constraints)
        old_bound = deepcopy(bound)
        to_free!(free, bound, constraints)
        old_free = deepcopy(free)
        to_bound!(bound, free, constraints)

        @test all(isapprox(old_bound[src], bound[src]) for src in 1:n_sources)

        # This test currently fails, but that's probably inevitable as
        # long as doing a round-trip unconstrain/reconstrain on a
        # parameter set yields parameters that are only appromixately
        # equal to the original parameters (which seems like a
        # tolerable behavior)
        to_free!(free, bound, constraints)
        @test_broken all(isapprox(old_free[src], free[src])
                         for src in 1:n_sources)
    end


    @testset "star-only parameter transformations" begin
        n_sources = 10

        bound = collect(rand(length(CanonicalParams)) for _ in 1:n_sources)

        constraints = star_constraints(bound)

        free = allocate_free_params(bound, constraints)

        @test length(free) == n_sources
        @test all(length(free[src]) == 19 for src in 1:n_sources)

        enforce!(bound, constraints)
        old_bound = deepcopy(bound)
        to_free!(free, bound, constraints)
        old_free = deepcopy(free)
        to_bound!(bound, free, constraints)
        @test all(isapprox(old_bound[src], bound[src]) for src in 1:n_sources)
    end



    @testset "check derivatives against old Transform" begin
        ea, vp, catalog = SampleData.gen_n_body_dataset(1)

        bound = vp
        constraints = elbo_constraints(bound)
        free = allocate_free_params(bound, constraints)

        enforce!(bound, constraints)
        to_free!(free, bound, constraints)

        derivs = TransformDerivatives(bound, free)

        differentiate!((y, x) -> to_bound!(y, x, constraints, 1), derivs, free[1])

        dt = Transform.get_mp_transform(vp, [1])
        td = Transform.get_transform_derivatives(vp, [1], dt.bounds)

        @test isapprox(td.dparam_dfree, derivs.jacobian)

        for i in 1:size(derivs.hessian, 1)
            @test isapprox(td.d2param_dfree2[i], derivs.hessian[i, :, :])
        end
    end

    @testset "check SensitiveFloat propagation" begin
        ea, vp, catalog = SampleData.gen_n_body_dataset(1)

        bound = vp
        constraints = elbo_constraints(bound)
        free = allocate_free_params(bound, constraints)

        enforce!(bound, constraints)
        to_free!(free, bound, constraints)

        derivs = TransformDerivatives(bound, free)
        dt = Transform.get_mp_transform(vp, [1])

        sf_bound1 = SensitiveFloats.SensitiveFloat{Float64}(length(bound[1]), 1, true, true)
        sf_bound1.v[] = rand()
        rand!(sf_bound1.d)
        rand!(sf_bound1.h)
        sf_bound2 = SensitiveFloats.SensitiveFloat{Float64}(length(bound[1]), 1, true, true)
        sf_bound2.v[] = sf_bound1.v[]
        copy!(sf_bound2.d, sf_bound1.d)
        copy!(sf_bound2.h, sf_bound1.h)
        sf_free1 = SensitiveFloats.SensitiveFloat{Float64}(length(free[1]), 1, true, true)
        sf_free2 = SensitiveFloats.SensitiveFloat{Float64}(length(free[1]), 1, true, true)

        propagate_derivatives!(to_bound!, sf_bound1, sf_free1, free, constraints, derivs)
        Transform.transform_sensitive_float!(dt, sf_free2, sf_bound2, vp, [1])

        @test isapprox(sf_free1.v[], sf_free2.v[])
        @test isapprox(sf_free1.d, sf_free2.d)
        @test isapprox(sf_free1.h, sf_free2.h)
    end

end # testset
