#!/usr/bin/env julia

using FactCheck


include("../src/newton_trust_region.jl")


facts("solve_tr_subproblem! finds the minimium") do
    for i in 1:1000
        n = rand(1:10)
        gr = randn(n)
        H = randn(n, n)
        H = H * H' + 5 * eye(n)
        s = zeros(n)
        m, interior = solve_tr_subproblem!(gr, H, 1., s, max_iters=100)

        model(s2) = (gr' * s2)[] + .5 * (s2' * H * s2)[]
        @fact model(s) --> less_than(model(zeros(n)) + 1e-8)  # origin

        for j in 1:10
            bad_s = rand(n)
            bad_s ./= norm(bad_s)  # boundary
            @fact model(s) --> less_than(model(bad_s) + 1e-8)
            bad_s .*= rand()  # interior
            @fact model(s) --> less_than(model(bad_s) + 1e-8)
        end
    end
end
