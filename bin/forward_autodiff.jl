using Celeste
using CelesteTypes

using DataFrames
using SampleData

using ForwardDiff
using DualNumbers
import Transform

blob, mp, body = gen_sample_star_dataset();
transform = Transform.pixel_rect_transform;
#omitted_ids = [ids_free.k[:], ids_free.c2[:], ids_free.r2];
omitted_ids = Int64[];
kept_ids = setdiff(1:length(ids_free), omitted_ids)
x0 = transform.vp_to_vector(mp.vp, omitted_ids);
x0_dual = Dual{Float64}[ Dual{Float64}(x0[i], 0.) for i = 1:length(x0) ]

mp_dual = ModelParams(convert(Array{Array{Dual{Float64}, 1}, 1}, mp.vp), mp.pp, mp.patches, mp.tile_width);


include("src/KL.jl")
include("src/ElboDeriv.jl")
function objective(x::Array{Float64})
    # Evaluate in the constrained space and then unconstrain again.
    println("Float")
    x_dual = Dual{Float64}[ Dual{Float64}(x[i], 0.) for i = 1:length(x) ]
    transform.vector_to_vp!(x_dual, mp_dual.vp, omitted_ids)
    elbo = ElboDeriv.elbo(blob, mp_dual)
    elbo.v
end
function objective(x::Array{Dual{Float64}})
    # Evaluate in the constrained space and then unconstrain again.
    println("Dual")
    transform.vector_to_vp!(x, mp_dual.vp, omitted_ids)
    elbo = ElboDeriv.elbo(blob, mp_dual)
    elbo.v
end

objective(x0)
objective(x0_dual)

objective_grad = ForwardDiff.forwarddiff_gradient(objective, Float64, fadtype=:dual; n=length(x0));
g_fd = objective_grad(x0)
celeste_elbo = transform.transform_sensitive_float(ElboDeriv.elbo(blob, mp), mp);
hcat(g_fd, celeste_elbo.d[kept_ids])
g_fd - celeste_elbo.d[kept_ids]


