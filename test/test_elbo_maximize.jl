module ElboMaximizeTests

using Celeste.DeterministicVI
using Celeste.DeterministicVI: ElboMaximize, ConstraintTransforms
using Base.Test
using Calculus

include(joinpath(dirname(@__FILE__), "Synthetic.jl"))
include(joinpath(dirname(@__FILE__), "SampleData.jl"))

ea, vp, _ = SampleData.gen_two_body_dataset()

cfg = ElboMaximize.Config(ea, vp)
ElboMaximize.enforce_references!(ea, vp, cfg)
ConstraintTransforms.enforce!(cfg.bound_params, cfg.constraints)
ConstraintTransforms.to_free!(cfg.free_params, cfg.bound_params, cfg.constraints)

x = ElboMaximize.to_vector(cfg.free_params)

# testing the objective
f = ElboMaximize.Objective(ea, vp, cfg)
@test f(x) == -(DeterministicVI.elbo(ea, vp).v[])

# testing the gradient
actual_grad = zeros(x)
g! = ElboMaximize.Gradient(ea, vp, cfg)
g!(x, actual_grad)
finite_diff_grad = Calculus.gradient(f, x)
@test isapprox(finite_diff_grad, actual_grad, atol=1e-5)

# testing the Hessian-vector product
actual_hvp, v = zeros(x), ones(x)
hvp! = ElboMaximize.HessianVectorProduct(ea, vp, cfg)
hvp!(x, v, actual_hvp)
g = y -> (z = zeros(y); g!(y, z); z)
finite_diff_hvp = Calculus.jacobian(g, x, :central) * v
@test all(isapprox.(finite_diff_hvp ./ actual_hvp, 1.0, atol=1e-5))
@test isapprox(finite_diff_hvp, actual_hvp, atol=0.05)

end # module
