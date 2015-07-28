using Celeste
using CelesteTypes

using DataFrames
using SampleData

using ForwardDiff
using DualNumbers
import Transform
import Optim

include("src/interpolating_linesearch.jl")

# Note that the u hessians are no good.
#omitted_ids = Int64[ids_free.u, ids_free.k[:], ids_free.c2[:], ids_free.r2];
omitted_ids = ids_free.u;

# Optimize only the star parameters.
# omitted_ids = reduce(vcat, [gal_ids.(name) for name in names(gal_ids)]);
# omitted_ids = union(omitted_ids, ids_free.u)
# omitted_ids = union(omitted_ids, ids_free.a)
# omitted_ids = union(omitted_ids, ids_free.c1[:,2])
# omitted_ids = union(omitted_ids, ids_free.c2[:,2])
# omitted_ids = union(omitted_ids, ids_free.r1[2])
# omitted_ids = union(omitted_ids, ids_free.r2[2])
# omitted_ids = union(omitted_ids, ids_free.k[:,2])
# omitted_ids = unique(omitted_ids)

kept_ids = setdiff(1:length(ids_free), omitted_ids)

#blob, mp_original, body = gen_sample_star_dataset()
#blob, mp_original, body = gen_sample_galaxy_dataset(perturb=true);
#blob, mp_original, body = gen_three_body_dataset(perturb=true); # Too slow.



transform = Transform.free_transform;

##############
mp_fit = deepcopy(mp_original);
x0 = transform.vp_to_vector(mp_fit.vp, omitted_ids);
x0_dual = Dual{Float64}[ Dual{Float64}(x0[i], 0.) for i = 1:length(x0) ];

obj_wrap = OptimizeElbo.ObjectiveWrapperFunctions(
    mp -> ElboDeriv.elbo(blob, mp), deepcopy(mp_original), transform, kept_ids, omitted_ids);

if false # Checks
    x_elbo_d = obj_wrap.f_grad(x0);
    g_fd = obj_wrap.f_ad_grad(x0);
    DataFrame(name=ids_free_names[kept_ids], elbo_d=x_elbo_d, ad_d=g_fd, diff=x_elbo_d - g_fd)
end

scale = -1.0
function elbo_scale_value(x)
    val = scale * obj_wrap.f_value(x)
    println("Elbo: $val")
    val
end
function elbo_scale_deriv!(x, grad)
    grad[:] = scale * obj_wrap.f_grad(x)
end
function elbo_scale_hess!(x, hess)
    hess[:, :] = scale * obj_wrap.f_ad_hessian(x)
end

elbo_grad = zeros(Float64, length(x0));
elbo_hess = zeros(Float64, length(x0), length(x0));

if false
    elbo_scale_value(x0)
    elbo_scale_deriv!(x0, elbo_grad)
    elbo_scale_hess!(x0, elbo_hess);

    hess_eig_val = eig(elbo_hess)[1];
    hess_eig_vec = eig(elbo_hess)[2];
    sort(hess_eig_val)
    eigs = DataFrame([ round(hess_eig_vec[:,i], 3) for i in sortperm(hess_eig_val)]);
    eigs[:name] = ids_free_names[kept_ids]
    for i = 1:length(kept_ids)
        println(sort(hess_eig_val)[i])
        println(eigs[[:name, symbol("x$i")]])
    end
end

#########################
# Newton's method by hand
function print_x_params(x::Vector{Float64})
    mp_copy = deepcopy(mp_original)
    transform.vector_to_vp!(x, mp_copy.vp, omitted_ids)
    print_params(mp_copy)
end


# Get a BFGS fit for comparison
mp_fit = deepcopy(mp_original)
iter_count, max_f, max_x, ret = OptimizeElbo.maximize_f(ElboDeriv.elbo, blob, mp_fit, Transform.free_transform, omitted_ids=omitted_ids);
fit_v = ElboDeriv.elbo(blob, mp_fit).v;

# Stuff:
hess_reg = 0.0;
max_iters = 5;

d = Optim.DifferentiableFunction(elbo_scale_value, elbo_scale_deriv!);
x_old = deepcopy(x0);
x_new = deepcopy(x_old);
gr_new = zeros(Float64, length(x_old));
iter = 1
f_val = elbo_scale_value(x_new);

f_vals = zeros(Float64, max_iters);
cumulative_iters = zeros(Int64, max_iters);
x_vals = [ zeros(Float64, length(x_old)) for iter=1:max_iters ];

rho = 2.0;
max_backstep = 20;

# warm start with BFGS
mp_start = deepcopy(mp_original)
start_iter_count, start_f, x_start = OptimizeElbo.maximize_f(ElboDeriv.elbo, blob, mp_start, Transform.free_transform, omitted_ids=omitted_ids, ftol_abs=10);
obj_wrap.state.f_evals = start_iter_count;
x_new = deepcopy(x_start); # For quick restarts while debugging
for iter in 1:max_iters
    println("-------------------$iter")
    x_old = deepcopy(x_new);
    #x_direction = -1e-6 * gr_new;
    elbo_scale_hess!(x_new, elbo_hess);
    hess_ev = eig(elbo_hess)[1]
    min_ev = minimum(hess_ev)
    max_ev = maximum(hess_ev)
    println("========= Eigenvalues: $(max_ev), $(min_ev)")
    if min_ev < 0
        println("========== Warning -- non-convex, $(min_ev)")
        elbo_hess += eye(length(x_new)) * abs(min_ev)
        hess_ev = eig(elbo_hess)[1]
        min_ev = minimum(hess_ev)
        max_ev = maximum(hess_ev)
        println("========= New eigenvalues: $(max_ev), $(min_ev)")
    end
    if abs(max_ev) / abs(min_ev) > 1e6
        println("Regularizing hessian")
        elbo_hess += eye(length(x_new)) * (abs(max_ev) / 1e6)
        hess_ev = eig(elbo_hess)[1]
        min_ev = minimum(hess_ev)
        max_ev = maximum(hess_ev)
        println("========= New eigenvalues: $(max_ev), $(min_ev)")
    end
    gr_new = zeros(Float64, length(x_old));
    elbo_scale_deriv!(x_old, gr_new);
    x_direction = -(elbo_hess \ gr_new)
    alpha = 1.0;
    backsteps = 0;
    new_val = elbo_scale_value(x_old + alpha * x_direction)
    while isnan(new_val)
        alpha /= rho;
        println("Backstepping: ")
        backsteps += 1;
        if backsteps > max_backstep
            error("Not a descent direction.")
        end
        new_val = elbo_scale_value(x_old + alpha * x_direction)
    end
    x_direction = alpha * x_direction
    #println(DataFrame(name=ids_free_names[kept_ids], grad=gr_new, hess=diag(elbo_hess), p=x_direction))
    lsr = Optim.LineSearchResults(Float64); # Not used
    c = -1.; # Not used
    mayterminate = true; # Not used
    #if backsteps == 0
    interpolating_linesearch!(d, x_old, x_direction,
                              x_new, gr_new,
                              lsr, c, mayterminate;
                              c1 = 1e-4,
                              c2 = 0.9,
                              rho = 2.0, verbose=false);
    print_x_params(x_new)
    # else
    #     a_star, f_up, g_up = zoom(0., 1.0,
    #                               dot(gr_new, x_direction), new_val,
    #                               elbo_scale_value, elbo_scale_deriv!,
    #                               x_old, x_direction, x_new, gr_new, verbose=true)
    # end
    this_f_val = elbo_scale_value(x_new)
    f_vals[iter] = this_f_val;
    x_vals[iter] = deepcopy(x_new)
    cumulative_iters[iter] = obj_wrap.state.f_evals
    println(">>>>>>  Current value after $(obj_wrap.state.f_evals) evaluations: $(this_f_val) (BFGS got $(-fit_v))")
    mp_nm = deepcopy(mp_original);
    transform.vector_to_vp!(x_new, mp_nm.vp, omitted_ids);
    println(ElboDeriv.get_brightness(mp_nm))
    println("\n\n")
end

println("Newton objective - BFGS objective (higher is better)")
((-f_vals) - fit_v) / abs(fit_v) # f_vals are negative because it's minimization

println("Cumulative fuction evaluation ratio:")
cumulative_iters ./ iter_count

f_vals
diff(f_vals) ./ f_vals[1:(end-1)]
minimum(f_vals)

[ x ./ [x_vals[1] - x_vals[end]] for x in diff(x_vals) ]

mp_nm = deepcopy(mp_original);
transform.vector_to_vp!(x_new, mp_nm.vp, omitted_ids);


print_params(mp_nm, mp_fit, mp_original)

get_brightness(mp_nm)
get_brightness(mp_fit)
get_brightness(mp_original)


# ################
# # Newton out of the box takes too many bad steps
# optim_res0 = Optim.optimize(elbo_scale_value,
#                              elbo_scale_deriv!,
#                              elbo_scale_hess!,
#                              x0, method=:newton,
#                              show_trace=true, ftol=1e-6, xtol=0.0, grtol=1e-4, iterations=30)

