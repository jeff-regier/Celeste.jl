using Celeste
using CelesteTypes

using DataFrames
using SampleData

using ForwardDiff
using DualNumbers
import Transform
import Optim

blob, mp_original, body = gen_sample_star_dataset();
mp = deepcopy(mp_original);
transform = Transform.free_transform;
# Note that the u hessians are no good.

#omitted_ids = Int64[ids_free.u, ids_free.k[:], ids_free.c2[:], ids_free.r2];
#omitted_ids = ids_free.u;

# Optimize only the star parameters.
omitted_ids = reduce(vcat, [gal_ids.(name) for name in names(gal_ids)]);
omitted_ids = union(omitted_ids, ids_free.u)
omitted_ids = union(omitted_ids, ids_free.a)
omitted_ids = union(omitted_ids, ids_free.c1[:,2])
omitted_ids = union(omitted_ids, ids_free.c2[:,2])
omitted_ids = union(omitted_ids, ids_free.c1[:,2])
omitted_ids = union(omitted_ids, ids_free.r1[2])
omitted_ids = union(omitted_ids, ids_free.r2[2])
omitted_ids = unique(omitted_ids)

kept_ids = setdiff(1:length(ids_free), omitted_ids)
eps = 1e-9
mp.vp[1][ids.a] = [ 1.0 - eps, eps ]

mp_fit = deepcopy(mp_original);
mp_fit.vp[1][ids.a] = [ 1.0 - eps, eps ]
OptimizeElbo.maximize_f(ElboDeriv.elbo, blob, mp_fit, Transform.free_transform, omitted_ids=omitted_ids);


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

hess_reg = 0.0;
scale = -1.0;
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
    # Normally we maximize, so the hessian should be negative definite.
    scale * (elbo_hess - hess_reg * eye(length(x)))
end

function get_elbo_hessian!(x::Array{Float64}, hess)
    hess[:,:] = get_elbo_hessian(x)
end

function get_id_hessian!(x::Array{Float64}, hess)
    hess[:, :] = -1.0 * scale * eye(Float64, length(x))
end

function get_elbo_derivative!(x::Array{Float64}, grad)
    grad[:] = scale * Float64[ real(x_val) for x_val in elbo_deriv(x)[kept_ids] ]
end

function get_elbo_value(x::Array{Float64})
    elbo_val = scale * real(elbo_value(x))
    println("elbo val: $elbo_val")
    elbo_val
end

if false
    # Newton's method doesn't work very well out of the box -- lots of bad steps.
    optim_res0 = Optim.optimize(get_elbo_value,
                                 get_elbo_derivative!,
                                 get_elbo_hessian!,
                                 x0, method=:newton, show_trace=true, ftol=1e-6, xtol=0.0, grtol=1e-4, iterations=30)

    # Try first steps with an identity gradient then NM.
    optim_res0 = Optim.optimize(get_elbo_value,
                                 get_elbo_derivative!,
                                 get_id_hessian!,
                                 x0, method=:newton, show_trace=true, ftol=1e-6, xtol=0.0, grtol=1e-4, iterations=30)

    x1 = optim_res0.minimum;
    optim_res1 = Optim.optimize(get_elbo_value,
                                 get_elbo_derivative!,
                                 get_elbo_hessian!,
                                 x1, method=:newton, show_trace=true, ftol=1e-6, xtol=0.0, grtol=1e-4, iterations=30)
    x = optim_res1.minimum;
end


##########
max_iters = 30;

d = Optim.DifferentiableFunction(get_elbo_value, get_elbo_derivative!);
x_old = deepcopy(x0);
x_new = deepcopy(x_old);
gr_new = zeros(Float64, length(x_old));
get_elbo_derivative!(x_old, gr_new);
iter = 1
f_val = get_elbo_value(x_new);

elbo_hess = zeros(Float64, length(kept_ids), length(kept_ids));
get_elbo_hessian!(x_new, elbo_hess);
f_vals = zeros(Float64, max_iters)
x_vals = [ zeros(Float64, length(x_old)) for iter=1:max_iters ]
println(DataFrame(name=ids_free_names[kept_ids], grad=gr_new, hess=diag(elbo_hess)))

hess_eig_val = eig(elbo_hess)[1];
hess_eig_vec = eig(elbo_hess)[2];
sort(hess_eig_val)
# eigs = DataFrame([ round(hess_eig_vec[:,i], 3) for i in sortperm(hess_eig_val)]);
# eigs[:name] = ids_free_names[kept_ids]
# for i = 1:length(kept_ids)
#     println(sort(hess_eig_val)[i])
#     println(eigs[[:name, symbol("x$i")]])
# end

rho = 2.0;
max_backstep = 20;
include("src/interpolating_linesearch.jl")
for iter in 1:max_iters
    println("-------------------$iter")
    x_old = deepcopy(x_new);
    #x_direction = -1e-6 * gr_new;
    get_elbo_hessian!(x_new, elbo_hess);
    hess_ev = eig(elbo_hess)[1]
    min_ev = minimum(hess_ev)
    max_ev = maximum(hess_ev)
    if min_ev < 0
        println("========== Warning -- non-convex, $(min_ev)")
        elbo_hess += eye(length(x_new)) * abs(min_ev)
        hess_ev = eig(elbo_hess)[1]
        min_ev = minimum(hess_ev)
        max_ev = maximum(hess_ev)
    end
    println("========= Eigenvalues: $(max_ev), $(min_ev)")
    if abs(max_ev) / abs(min_ev) > 1e3
        println("Regularizing hessian")
        elbo_hess += eye(length(x_new)) * (abs(max_ev) / 1e6)
        hess_ev = eig(elbo_hess)[1]
        min_ev = minimum(hess_ev)
        max_ev = maximum(hess_ev)
    end
    println("========= Eigenvalues: $(max_ev), $(min_ev)")
    x_direction = -(elbo_hess \ gr_new);
    alpha = 1.0;
    backsteps = 0;
    while isnan(get_elbo_value(x_old + alpha * x_direction))
        alpha /= rho;
        println("Backstepping: ")
        backsteps += 1;
        if backsteps > max_backstep
            error("Not a descent direction.")
        end
    end
    x_direction = alpha * x_direction
    gr_new = zeros(Float64, length(x_old));
    get_elbo_derivative!(x_old, gr_new);
    #println(DataFrame(name=ids_free_names[kept_ids], grad=gr_new, hess=diag(elbo_hess), p=x_direction))
    lsr = Optim.LineSearchResults(Float64); # Not used
    c = -1.; # Not used
    mayterminate = true; # Not used
    interpolating_linesearch!(d, x_old, x_direction,
                              x_new, gr_new,
                              lsr, c, mayterminate;
                              c1 = 1e-4,
                              c2 = 0.9,
                              rho = 2.0, verbose=false);
    f_vals[iter] = get_elbo_value(x_new);
    x_vals[iter] = deepcopy(x_new)
end

f_vals
diff(f_vals) ./ f_vals[1:(end-1)]
minimum(f_vals)

[ x ./ [x_vals[1] - x_vals[end]] for x in diff(x_vals) ]


last_f_vals = deepcopy(f_vals)

transform.vector_to_vp!(x_new, mp.vp, omitted_ids)


function get_brightness(mp::ModelParams)
    brightness = [ElboDeriv.SourceBrightness(mp.vp[s]) for s in 1:mp.S];
    brightness_vals = [ Float64[b.E_l_a[i, j].v for
        i=1:size(b.E_l_a, 1), j=1:size(b.E_l_a, 2)] for b in brightness]
    brightness_vals
end

function show_mp(mp_show)
    for var_name in names(ids)
        println(var_name)
        for s in 1:mp_show.S
            println(s, ":\n", mp_show.vp[s][ids.(var_name)])
        end
    end
end

show_mp(mp)
show_mp(mp_original)

get_brightness(mp)
get_brightness(mp_fit)
get_brightness(mp_original)

fit_v = ElboDeriv.elbo(blob, mp_fit).v;
((-f_vals) - fit_v) / abs(fit_v) # f_vals are negative because it's minimization
(ElboDeriv.elbo(blob, mp).v - fit_v) / abs(fit_v)