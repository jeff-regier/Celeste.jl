using Celeste
using CelesteTypes

using DataFrames
using SampleData

using ForwardDiff
using DualNumbers
import Transform
import Optim



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

function get_dual_mp(mp::ModelParams{Float64})
    ModelParams(convert(Array{Array{Dual{Float64}, 1}, 1}, mp.vp),
                mp.pp, mp.patches, mp.tile_width)
end


function get_autodiff_funcs(mp, kept_ids, omitted_ids, transform, f::Function)

    mp_dual = get_dual_mp(mp);

    function f_objective(x_dual::Array{Dual{Float64}})
        # Evaluate in the constrained space and then unconstrain again.
        transform.vector_to_vp!(x_dual, mp_dual.vp, omitted_ids)
        f_res = f(mp_dual)
        res = transform.transform_sensitive_float(f_res, mp_dual)
        res
    end

    function f_objective(x::Array{Float64})
        # Evaluate in the constrained space and then unconstrain again.
        x_dual = Dual{Float64}[ Dual{Float64}(x[i], 0.) for i = 1:length(x) ]
        f_objective(x_dual)
    end

    function f_value(x)
        @assert length(x) == length(kept_ids)
        f_objective(x).v
    end

    function f_deriv(x)
        @assert length(x) == length(kept_ids)
        f_objective(x).d[:]
    end

    function f_deriv(x::Vector{Float64})
        @assert length(x) == length(kept_ids)
        real(f_objective(x).d[:])
    end

    function f_hessian(x::Array{Float64})
        @assert length(x) == length(kept_ids)
        k = length(kept_ids)
        hess = zeros(Float64, k, k);
        x_dual = Dual{Float64}[ Dual{Float64}(x[i], 0.) for i = 1:k ]
        print("Getting Hessian ($k components): ")
        for index in 1:k
            print(".")
            x_dual[index] = Dual(x[index], 1.)
            deriv = f_deriv(x_dual)[kept_ids]
            hess[:, index] = Float64[ epsilon(x_val) for x_val in deriv ]
            x_dual[index] = Dual(x[index], 0.)
        end
        print("Done.\n")
        hess
    end

    f_ad_grad = ForwardDiff.forwarddiff_gradient(f_value, Float64, fadtype=:dual; n=length(kept_ids));

    f_ad_grad, f_value, f_deriv, f_objective, f_hessian
end



# Note that the u hessians are no good.
#omitted_ids = Int64[ids_free.u, ids_free.k[:], ids_free.c2[:], ids_free.r2];
#omitted_ids = ids_free.u;

# Optimize only the star parameters.
omitted_ids = reduce(vcat, [gal_ids.(name) for name in names(gal_ids)]);
omitted_ids = union(omitted_ids, ids_free.u)
omitted_ids = union(omitted_ids, ids_free.a)
omitted_ids = union(omitted_ids, ids_free.c1[:,2])
omitted_ids = union(omitted_ids, ids_free.c2[:,2])
omitted_ids = union(omitted_ids, ids_free.r1[2])
omitted_ids = union(omitted_ids, ids_free.r2[2])
omitted_ids = union(omitted_ids, ids_free.k[:,2])
omitted_ids = unique(omitted_ids)

kept_ids = setdiff(1:length(ids_free), omitted_ids)


using Base.Test
function test_elbo_invariance_to_a()
    # Changing this line to 1000 causes the test to fail.
    fluxes = [2.47122, 1.832, 4.0, 5.9192, 9.12822] * 1000
    ce = CatalogEntry([7.2,8.3], false, fluxes, fluxes, 0.5, .7, pi/4, .5)
    blob0 = SDSS.load_stamp_blob(dat_dir, "164.4311-39.0359")
    for b in 1:5
        blob0[b].H, blob0[b].W = 20, 23
        blob0[b].wcs = WCS.wcs_id
    end
    blob = Synthetic.gen_blob(blob0, [ce,])

    mp = ModelInit.cat_init([ce,])
    mp.vp[1][ids.a] = [ 0.8, 0.2 ]
    omitted_ids = [ids_free.a, ids_free.r2[:], ids_free.c2[:], ids_free.e_dev]
    OptimizeElbo.maximize_f(ElboDeriv.elbo, blob, mp,
        Transform.pixel_rect_transform, omitted_ids=omitted_ids)

    mp2 = ModelInit.cat_init([ce,])
    mp2.vp[1][ids.a] = [ 0.2, 0.8 ]
    OptimizeElbo.maximize_f(ElboDeriv.elbo, blob, mp2,
        Transform.pixel_rect_transform, omitted_ids=omitted_ids)

    mp.vp[1][ids.a] = [ 0.5, 0.5 ]
    mp2.vp[1][ids.a] = [ 0.5, 0.5 ]
    @test_approx_eq_eps ElboDeriv.elbo(blob, mp).v ElboDeriv.elbo(blob, mp2).v 1

    for i in setdiff(1:length(CanonicalParams), ids.a) #skip a
        @test_approx_eq_eps mp.vp[1][i] / mp2.vp[1][i] 1. 0.1
    end
end
#test_elbo_invariance_to_a()



transform = Transform.free_transform;
if false
    # This is strongly affected by a for some reason
    blob, mp_original, body = gen_sample_star_dataset(perturb=false);

    # blob0 = SDSS.load_stamp_blob(dat_dir, "164.4311-39.0359")
    # for b in 1:5
    #     blob0[b].H, blob0[b].W = 20, 23
    #     blob0[b].wcs = WCS.wcs_id
    # end

    # one_body = [sample_ce([10.1, 12.2], true),]
    # blob = Synthetic.gen_blob(blob0, one_body)
    # mp = ModelInit.cat_init(one_body)
elseif false
    blob, mp_original, body = gen_sample_galaxy_dataset(perturb=false);
else
    # Load an example from test_optimization
    # This was originally a galaxy dataset.
    fluxes = [2.47122, 1.832, 4.0, 5.9192, 9.12822] * 100;
    is_star = false

    ce = CatalogEntry([7.2,8.3], is_star, fluxes, fluxes, 0.5, .7, pi/4, .5);
    #ce = SampleData.sample_ce([7.2,8.3], is_star);
    blob0 = SDSS.load_stamp_blob(dat_dir, "164.4311-39.0359");
    for b in 1:5
        blob0[b].H, blob0[b].W = 20, 23;
        blob0[b].wcs = WCS.wcs_id;
    end
    blob = Synthetic.gen_blob(blob0, [ce,]);
    mp_original = ModelInit.cat_init([ce,]);

    # Artificially change the galaxy parameters
    # mp_original.vp[1][ids.r1[2]] *= 2.0
    # mp_original.vp[1][ids.e_scale] *= 2.0
end

if false
    eps = 0.01
    eps_vec = linspace(eps, 1 - eps, 3);
    brightness_results = Array(Any, length(eps_vec));
    elbo_results = Array(Float64, length(eps_vec));
    mp_results = Array(Any, length(eps_vec));
    x_results = Array(Any, length(eps_vec));

    for i in 1:length(eps_vec)
        this_eps = eps_vec[i]
        mp_fit = deepcopy(mp_original);
        mp_fit.vp[1][ids.a] = [ 1.0 - this_eps, this_eps ]
        println("$i $(mp_fit.vp[1][ids.a])")
        iter_count, max_f, max_x, ret = OptimizeElbo.maximize_f(ElboDeriv.elbo, blob, mp_fit, Transform.free_transform, omitted_ids=omitted_ids, ftol_abs=1e-4);
        elbo_results[i] = ElboDeriv.elbo(blob, mp_fit).v;
        brightness_results[i] = get_brightness(mp_fit);
        mp_results[i] = deepcopy(mp_fit);
        x_results[i] = max_x
    end

    reduce(hcat, [ b[1][:,1] for b in brightness_results ])
end


##############
mp_fit = deepcopy(mp_original);
x0 = transform.vp_to_vector(mp_fit.vp, omitted_ids);
x0_dual = Dual{Float64}[ Dual{Float64}(x0[i], 0.) for i = 1:length(x0) ];

elbo_ad_grad, elbo_value, elbo_deriv, elbo_objective, elbo_hessian =
    get_autodiff_funcs(mp_fit, kept_ids, omitted_ids, Transform.free_transform,
                       mp -> ElboDeriv.elbo(blob, mp));

x_elbo_d = elbo_deriv(x0);
g_fd = elbo_ad_grad(x0);
DataFrame(name=ids_free_names[kept_ids], elbo_d=x_elbo_d[kept_ids], ad_d=g_fd, diff=x_elbo_d[kept_ids] - g_fd)

scale = -1.0
function elbo_scale_value(x)
    val = scale * real(elbo_value(x))
    println("Elbo: $val")
    val
end
function elbo_scale_deriv!(x, grad)
    grad[:] = scale * elbo_deriv(x)[kept_ids]
end
function elbo_scale_hess!(x, hess)
    hess[:, :] = scale * elbo_hessian(x)
end

elbo_grad = zeros(Float64, length(x0));
elbo_hess = zeros(Float64, length(x0), length(x0));

elbo_scale_value(x0)
elbo_scale_deriv!(x0, elbo_grad)
elbo_scale_hess!(x0, elbo_hess);

if false
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

# Get a BFGS fit for comparison
mp_fit = deepcopy(mp_original)
iter_count, max_f, max_x, ret = OptimizeElbo.maximize_f(ElboDeriv.elbo, blob, mp_fit, Transform.free_transform, omitted_ids=omitted_ids);
fit_v = ElboDeriv.elbo(blob, mp_fit).v;

# Stuff:
hess_reg = 0.0;
max_iters = 10;

d = Optim.DifferentiableFunction(elbo_scale_value, elbo_scale_deriv!);
x_old = deepcopy(x0);
x_new = deepcopy(x_old);
gr_new = zeros(Float64, length(x_old));
iter = 1
f_val = elbo_scale_value(x_new);

f_vals = zeros(Float64, max_iters);
x_vals = [ zeros(Float64, length(x_old)) for iter=1:max_iters ];

rho = 2.0;
max_backstep = 20;
include("src/interpolating_linesearch.jl")

# warm start with BFGS
mp_start = deepcopy(mp_original)
start_iter_count, start_f, x_new = OptimizeElbo.maximize_f(ElboDeriv.elbo, blob, mp_start, Transform.free_transform, omitted_ids=omitted_ids, ftol_abs=1000);
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
    while isnan(elbo_scale_value(x_old + alpha * x_direction))
        alpha /= rho;
        println("Backstepping: ")
        backsteps += 1;
        if backsteps > max_backstep
            error("Not a descent direction.")
        end
    end
    x_direction = alpha * x_direction
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
    this_f_val = elbo_scale_value(x_new)
    f_vals[iter] = this_f_val;
    x_vals[iter] = deepcopy(x_new)
    println(">>>>>>  Current value $(this_f_val) (BFGS got $(-fit_v))")
    mp_nm = deepcopy(mp_original);
    transform.vector_to_vp!(x_new, mp_nm.vp, omitted_ids);
    println(get_brightness(mp_nm))
    println("\n\n")
end


((-f_vals) - fit_v) / abs(fit_v) # f_vals are negative because it's minimization
(ElboDeriv.elbo(blob, mp_nm).v - fit_v) / abs(fit_v)

f_vals
diff(f_vals) ./ f_vals[1:(end-1)]
minimum(f_vals)

[ x ./ [x_vals[1] - x_vals[end]] for x in diff(x_vals) ]

mp_nm = deepcopy(mp_original);
transform.vector_to_vp!(x_new, mp_nm.vp, omitted_ids);


show_mp(mp_nm)
show_mp(mp_fit)
show_mp(mp_original)

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

