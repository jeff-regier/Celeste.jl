using Celeste
using CelesteTypes

using DataFrames
using SampleData

using ForwardDiff
using DualNumbers
import Transform
import Optim

blob, mp, body = gen_sample_star_dataset();

transform = Transform.free_transform;
#omitted_ids = Int64[ids_free.u, ids_free.k[:], ids_free.c2[:], ids_free.r2];
omitted_ids = Int64[ids_free.u]; # The u derivatives are no good.
kept_ids = setdiff(1:length(ids_free), omitted_ids)
x0 = transform.vp_to_vector(mp.vp, omitted_ids);
x0_dual = Dual{Float64}[ Dual{Float64}(x0[i], 0.) for i = 1:length(x0) ]

mp_dual = ModelParams(convert(Array{Array{Dual{Float64}, 1}, 1}, mp.vp), mp.pp, mp.patches, mp.tile_width);

function elbo_objective(x_dual::Array{Dual{Float64}})
    # Evaluate in the constrained space and then unconstrain again.
    transform.vector_to_vp!(x_dual, mp_dual.vp, omitted_ids)
    elbo_res = ElboDeriv.elbo(blob, mp_dual)
    res = transform.transform_sensitive_float(elbo_res, mp_dual)
end

function elbo_objective(x::Array{Float64})
    # Evaluate in the constrained space and then unconstrain again.
    x_dual = Dual{Float64}[ Dual{Float64}(x[i], 0.) for i = 1:length(x) ]
    elbo_objective(x_dual)
end

function elbo_value(x)
    elbo_objective(x).v
end

function elbo_deriv(x)
    elbo_objective(x).d[:]
end

elbo_objective(x0);
elbo_objective(x0_dual);

objective_grad = ForwardDiff.forwarddiff_gradient(elbo_value, Float64, fadtype=:dual; n=length(x0));
g_fd = objective_grad(x0);

celeste_elbo = transform.transform_sensitive_float(ElboDeriv.elbo(blob, mp), mp);
hcat(g_fd, celeste_elbo.d[kept_ids])
g_fd - celeste_elbo.d[kept_ids]


function get_elbo_hessian(x::Array{Float64})
    k = length(kept_ids)
    @assert k == length(x)
    elbo_hess = zeros(Float64, k, k);
    x_dual = Dual{Float64}[ Dual{Float64}(x[i], 0.) for i = 1:k ]
    for index in 1:k
        println("Getting Hessian -- $index of $k")
        x_dual[index] = Dual(x[index], 1.)
        deriv = elbo_deriv(x_dual)[kept_ids]
        elbo_hess[:, index] = Float64[ epsilon(x_val) for x_val in deriv ]
        x_dual[index] = Dual(x[index], 0.)
    end
    elbo_hess
end

elbo_hess = get_elbo_hessian(x0)

maximum(abs(elbo_hess - elbo_hess'))
lambda = eig(elbo_hess)[1]
maximum(abs(lambda)) / minimum(abs(lambda)) # Very badly conditioned!

function get_elbo_hessian!(x::Array{Float64}, hess)
    hess[:,:] = -get_elbo_hessian(x)
end

function get_elbo_derivative!(x::Array{Float64}, grad)
    grad[:] = -Float64[ real(x_val) for x_val in elbo_deriv(x)[kept_ids] ]
end

function get_elbo_value(x::Array{Float64})
    elbo_val = -real(elbo_value(x))
    println("elbo val: $elbo_val")
    elbo_val
end

# Newton's method doesn't work very well out of the box -- lots of bad steps.
optim_res = Optim.optimize(get_elbo_value,
                             get_elbo_derivative!,
                             get_elbo_hessian!,
                             x0, method=:newton, show_trace=true)
