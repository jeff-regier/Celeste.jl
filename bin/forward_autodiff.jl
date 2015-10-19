using Celeste
using CelesteTypes

using DataFrames
using SampleData

using ForwardDiff
using DualNumbers
import Transform
import Optim

using PyPlot

galaxy_ids = union(ids_free.c1[:,2],
                   ids_free.c2[:,2],
                   ids_free.r1[2],
                   ids_free.r2[2],
                   ids_free.k[:,2],
                   ids_free.e_dev, ids_free.e_axis,
                   ids_free.e_angle, ids_free.e_scale);

star_ids = union(ids_free.c1[:,1],
                 ids_free.c2[:,1],
                 ids_free.r1[1],
                 ids_free.r2[1],
                 ids_free.k[:,1]);

simulation = true
if simulation
    #blob, mp_original, body, tiled_blob = gen_sample_star_dataset(perturb=true);

    # The gen_sample_galaxy locations and brightnesses look off.
    #blob, mp_original, body = gen_sample_galaxy_dataset(perturb=false);
    blob, mp_original, body, tiled_blob = gen_three_body_dataset(perturb=true);
    tiled_blob, mp_original = ModelInit.initialize_celeste(blob, body, tile_width=30);
    transform = Transform.get_mp_transform(mp_original, loc_width=1.0);
else
    # An actual celestial body.
    field_dir = joinpath(dat_dir, "sample_field")
    run_num = "003900"
    camcol_num = "6"
    field_num = "0269"

    original_blob =
      SkyImages.load_sdss_blob(field_dir, run_num, camcol_num, field_num,
                            mask_planes=Set());

    original_cat_df =
      SDSS.load_catalog_df(field_dir, run_num, camcol_num, field_num);
    cat_loc = convert(Array{Float64}, original_cat_df[[:ra, :dec]]);
    original_cat_entries =
      SkyImages.convert_catalog_to_celeste(original_cat_df, original_blob);

    obj_cols = [:objid, :is_star, :is_gal, :psfflux_r, :compflux_r, :ra, :dec];
    sort(original_cat_df[original_cat_df[:is_gal] .== true, obj_cols],
        cols=:compflux_r, rev=true)
    sort(original_cat_df[original_cat_df[:is_gal] .== false, obj_cols],
        cols=:psfflux_r, rev=true)

    #objid = "1237662226208063541" # A bright star with bad pixels --
    # I think this has a neighbor, too.

    objid = "1237662226208063499" # A bright star
    #objid = "1237662226208063576" # A galaxy
    objid = "1237662226208063565" # A brightish star but with good pixels.

    tile_width = 30

    tiled_blob, mp_original_all =
      ModelInit.initialize_celeste(
        original_blob, original_cat_entries,
        tile_width=tile_width, fit_psf=false);

    function CountSources(tiled_blob::TiledBlob)
      sources = Int64[]
      for b=1:5
        tile_sources =
          SkyImages.local_sources(
            tiled_blob[b][1, 1], mp_original_all.patches[:,b], original_blob[b].wcs)
        sources = union(sources, tile_sources)
      end
      sources
    end

    function get_object_tile(objid::ASCIIString; tile_width=30)
      obj_row = original_cat_df[:objid] .== objid;
      obj_loc = Float64[original_cat_df[obj_row, :ra][1],
                        original_cat_df[obj_row, :dec][1]]
      obj_pix_loc = [WCS.world_to_pixel(original_blob[b].wcs, obj_loc) for b=1:5]
      obj_row_num = find(obj_row)[1]

      # Make a tile for the object
      tiled_blob =
        SkyImages.crop_blob_to_location(original_blob, tile_width, obj_loc);
      mp_original = ModelInit.initialize_model_params(
        tiled_blob, original_blob, original_cat_entries[obj_row]);
      transform = Transform.get_mp_transform(mp_original);
      sources = CountSources(tiled_blob)

      tiled_blob, mp_original, transform, sources
    end

    # Make sure we only got one source
    tiled_blob, mp_original, transform, sources = get_object_tile(objid);
    @assert(length(sources) == 1, "$tile_sources")
    @assert(sources == [obj_row_num], "$sources")

    [ tiled_blob[b][1,1].h_range for b=1:5]
end

# Look at overlapping objects
for b=1:5
  lengths = Int64[ length(s) for s in mp_original.tile_sources[b] ]
  println(hcat(unique(lengths), counts(lengths)))
end


# Try the Hessian
param_msg = ElboDeriv.ParameterMessage(mp_original);
ElboDeriv.update_parameter_message!(mp_original, param_msg);
mp = deepcopy(mp_original);
accum = zero_sensitive_float(CanonicalParams, Float64, mp.S);
elbo_val = ElboDeriv.elbo_likelihood!(tiled_blob, param_msg, mp, accum);

mp_dual = CelesteTypes.convert(ModelParams{DualNumbers.Dual}, mp);
param_msg_dual = ElboDeriv.ParameterMessage(mp_dual);
ElboDeriv.update_parameter_message!(mp_dual, param_msg_dual);

using ElboDeriv.ParameterMessage
using ElboDeriv.tile_likelihood!

@doc """
Use forward auto-differentiation to compute the Hessian.
""" ->
function elbo_hessian(tiled_blob::TiledBlob,
    param_msg::ParameterMessage{Dual{Float64}},
    mp::ModelParams{Dual{Float64}}; verbose=true)

  # Vectors of the row, column, and value of the Hessian entries.
  # The indices are tuples of (source, parameter) which will be
  # linearized later.
  hess_i = (Int64, Int64)[]
  hess_j = (Int64, Int64)[]
  hess_val = Float64[]

  mp.vp = param_msg.vp
  accum = zero_sensitive_float(CanonicalParams, Dual{Float64}, mp.S)
  for b in 1:5
    sbs = param_msg.sbs_vec[b]
    star_mcs = param_msg.star_mcs_vec[b]
    gal_mcs = param_msg.gal_mcs_vec[b]
    for tile in tiled_blob[b][:]
      verbose && println("Tile .... ")
      tile_sources = mp.tile_sources[b][tile.hh, tile.ww]

      verbose && println(tile_sources)
      # Get the hessian entries (s1, index1), (s2, index2)
      for s1 in tile_sources, index1=1:length(CanonicalParams)
        # Get the derivative of the gradient wrt (s1, index1)
        verbose && println(s1, " ", index1)
        @assert DualNumbers.epsilon(mp.vp[s1][index1]) == 0.0
        mp.vp[s1][index1] = Dual(DualNumbers.real(mp.vp[s1][index1]), 1.0)
        clear!(accum)
        tile_likelihood!(tile, tile_sources, mp, sbs, star_mcs, gal_mcs, accum);
        for s2 in tile_sources, index2=1:length(CanonicalParams)
          push!(hess_i, (s1, index1))
          push!(hess_j, (s2, index2))
          push!(hess_val, DualNumbers.epsilon(accum.d[index2, s2]))
        end
        mp.vp[s1][index1] = Dual(DualNumbers.real(mp.vp[s1][index1]), 0.0)
      end
    end
  end
  hess_i, hess_j, hess_val
end
hess_i, hess_j, hess_val = elbo_hessian(tiled_blob, param_msg_dual, mp_dual);




##############
# Get a BFGS fit
function bfgs_fit_params(mp_original::ModelParams, omitted_ids::Array{Int64})
  mp_bfgs = deepcopy(mp_original);
  iter_count, max_f, max_x, ret =
    OptimizeElbo.maximize_f(ElboDeriv.elbo, tiled_blob, mp_bfgs, transform,
                            omitted_ids=omitted_ids, verbose=true);
  mp_bfgs, iter_count, max_f
end


####################
# Newton's method

# For newton's method.  It doesn't actually converge, so this
# controls the number of steps.
max_iters = 10;

#include("../Optim.jl/src/Optim.jl"); include("src/OptimizeElbo.jl")
function newton_fit_params(mp_original::ModelParams, omitted_ids::Array{Int64})
  mp_optim = deepcopy(mp_original);
  iter_count, max_f, max_x, ret =
    OptimizeElbo.maximize_f_newton(
      ElboDeriv.elbo, tiled_blob, mp_optim, transform,
      omitted_ids=omitted_ids, verbose=true, max_iters=max_iters,
      hess_reg=0.0)
  mp_optim, iter_count, max_f, ret
end



########################
# Run NM on all isolated objects

# This one has a bounds problem.
objid = "1237662226208063744"
tiled_blob, mp_original, transform, sources = get_object_tile(objid);

nm_all_results = Dict()
for objid in original_cat_df[:objid]
  println("\n\n\n\n...... FITTING $objid")
  tiled_blob, mp_original, transform, sources = get_object_tile(objid);
  if length(sources) > 1
    println(sources, ", skipping because of extra sources")
    continue
  end
  println("fitting $objid")
  println(original_cat_df[original_cat_df[:objid] .== objid, obj_cols])

  mp_optim, iter_count, max_f, ret = newton_fit_params(mp_original, Int64[])
  nm_all_results[objid] = (mp_original, mp_optim, iter_count, max_f, ret)
end


for (objid, result) in nm_all_results
  mp_original = result[1];
  mp_optim = result[2];
  println("\n\n\n\n", objid)
  println(original_cat_df[original_cat_df[:objid] .== objid, obj_cols])
  #print_params(result[1], result[2])
  println([mp_optim.vp[1][ids.a]])
  println([mp_optim.vp[1][ids.e_scale]])

  println("Original brighntess:\n", ElboDeriv.get_brightness(mp_original)[1])
  println("Fit brightness:\n", ElboDeriv.get_brightness(mp_optim)[1])
end


objid = collect(keys(nm_all_results))[6]

result = nm_all_results[objid];
tiled_blob, mp_original, transform, num_sources = get_object_tile(objid);
mp_optim = result[2];
for b=1:5
  tile = tiled_blob[b][1,1];
  pred_pix = ElboDeriv.tile_predicted_image(tile, mp_original, b);
  fit_pred_pix = ElboDeriv.tile_predicted_image(tile, mp_optim, b);

  pix_loc = WCS.world_to_pixel(original_blob[b].wcs, mp_original.vp[1][ids.u])
  tile_loc = pix_loc - [minimum(tile.h_range) - 1, minimum(tile.w_range) - 1]

  PyPlot.figure()
  plot_index = 1

  PyPlot.subplot(1, 3, plot_index)
  plot_index += 1
  PyPlot.imshow(pred_pix)
  PyPlot.plot(tile_loc[1] - 1., tile_loc[2] - 1., "r+")
  PyPlot.title("Original predicted band $b")

  PyPlot.subplot(1, 3, plot_index)
  plot_index += 1
  PyPlot.imshow(fit_pred_pix)
  PyPlot.plot(tile_loc[1] - 1., tile_loc[2] - 1., "r+")
  PyPlot.title("Fit predicted band $b")

  PyPlot.subplot(1, 3, plot_index)
  plot_index += 1
  PyPlot.imshow(tiled_blob[b][1,1].pixels)
  PyPlot.plot(tile_loc[1] - 1., tile_loc[2] - 1., "r+")
  PyPlot.title("Actual band $b")
end
#PyPlot.close("all")



# Show the original image near an object
pixel_width = 10
pixel_crop = 1000

obj_row = original_cat_df[:objid] .== objid;
obj_loc = Float64[original_cat_df[obj_row, :ra][1],
                  original_cat_df[obj_row, :dec][1]]
obj_pix_loc = [WCS.world_to_pixel(original_blob[b].wcs, obj_loc) for b=1:5]
obj_row_num = find(obj_row)[1]

original_cat_df[obj_row, obj_cols]
PyPlot.figure()
for b=1:5
  h_range =
    int(obj_pix_loc[b][1] - pixel_width):int(obj_pix_loc[b][1] + pixel_width)
  w_range =
    int(obj_pix_loc[b][2] - pixel_width):int(obj_pix_loc[b][2] + pixel_width)

  pixels = original_blob[b].pixels[h_range, w_range]
  pixels[pixels .> pixel_crop] = pixel_crop

  PyPlot.subplot(150 + b)
  PyPlot.imshow(pixels)
  PyPlot.plot(
    obj_pix_loc[b][1] - minimum(h_range),
    obj_pix_loc[b][2] - minimum(w_range), "r+")
  PyPlot.title("Band $b")
end

















#####################
# Older code

mp_bfgs_both_optim, bfgs_iter_count, max_f = bfgs_fit_params(mp_original, Int64[])
bfgs_v = ElboDeriv.elbo(tiled_blob, mp_bfgs_both_optim).v;
# 1014 iters


nm_results_both_optim, nm_iter_count, max_f, nm_ret =
    newton_fit_params(mp_original, Int64[]);

values = Float64[ s.value for s in nm_ret.trace.states ]
xs = [ s.metadata["x"] for s in nm_ret.trace.states ]
as = Float64[ x[ids_free.a] for x in xs ]

start = 1
i = 1
for i=1:length(ids_free_names)
  PyPlot.figure()
  plot(start:length(xs), [ x[i] for x in xs], "k.")
  PyPlot.title(ids_free_names[i])
end
PyPlot.close("all")

deltas = Float64[ s.metadata["delta"] for s in nm_ret.trace.states ]
plot(start:length(deltas), log(deltas)[start:end], "k.")
plot(start:length(values), log(values)[start:end], "r.")
plot(start:length(as), as[start:end], "b.")


nm_v = ElboDeriv.elbo(tiled_blob, nm_results_both_optim).v;

ElboDeriv.get_brightness(nm_results_both_optim)
ElboDeriv.get_brightness(mp_bfgs_both_optim)
ElboDeriv.get_brightness(mp_original)
print_params(nm_results_both_optim, mp_bfgs_both_optim, mp_original)
println("Newton elbo: $(nm_v) BFGS elbo: $(bfgs_v)")
println("Newton iters: $(nm_iter_count) BFGS iters: $(bfgs_iter_count)")


#############################
# Explore fitting only one object at a time.

function fit_only_type!(obj_type::Symbol, mp::ModelParams)
  valid_types = Symbol[:star, :galaxy, :both, :a]
  if !any(obj_type .== valid_types)
    error("obj_type must be in $(valid_types)")
  end
  epsilon = 0.006
  if obj_type == :star
    for s=1:mp.S
        mp.vp[s][ids.a] = [ 1.0 - epsilon, epsilon ]
    end
    omitted_ids = sort(unique(union(galaxy_ids, ids_free.a, ids_free.u)));
  elseif obj_type == :galaxy
    for s=1:mp.S
        mp.vp[s][ids.a] = [ epsilon, 1.0 - epsilon ]
    end
    omitted_ids = sort(unique(union(star_ids, ids_free.a, ids_free.u)));
  elseif obj_type == :both
    omitted_ids = Int64[];
  elseif obj_type == :a
    omitted_ids = setdiff(1:length(ids_free), ids_free.a)
    for s=1:mp.S
        mp.vp[s][ids.a] = [ 0.5, 0.5 ]
    end
  else
    error("obj_type must be in $(valid_types)")
  end
  omitted_ids
end



function fit_type(obj_type::Symbol, mp_original::ModelParams, fit_fun::Function)
  mp_type = deepcopy(mp_original)
  omitted_ids = fit_only_type!(obj_type, mp_type);
  mp_type, iter_count, max_f = fit_fun(mp_type, omitted_ids)
  mp_type, iter_count, max_f
end

function combine_star_gal(mp_star::ModelParams, mp_gal::ModelParams)
  mp_combined = deepcopy(mp_original);
  for s=1:mp_combined.S
    mp_combined.vp[s][galaxy_ids] = mp_gal.vp[s][galaxy_ids]
    mp_combined.vp[s][star_ids] = mp_gal.vp[s][star_ids]
    mp_combined.vp[s][ids.a] = [0.5, 0.5]
  end
  mp_combined
end



# Try fitting one type at a time.
bfgs_results = Dict()
nm_results = Dict()

nm_results[:star], nm_star_iters, nm_star_v =
  fit_type(:star, mp_original, newton_fit_params)
bfgs_results[:star], bfgs_star_iters, bfgs_star_v =
  fit_type(:star, mp_original, bfgs_fit_params)

nm_results[:galaxy], nm_gal_iters, nm_gal_v =
  fit_type(:galaxy, mp_original, newton_fit_params)
bfgs_results[:galaxy], bfgs_gal_iters, bfgs_gal_v =
  fit_type(:galaxy, mp_original, bfgs_fit_params)

println(nm_star_v, ", ", nm_star_iters)
println(bfgs_star_v, ", ", bfgs_star_iters)

println(nm_gal_v, ", ", nm_gal_iters)
println(bfgs_gal_v, ", ", bfgs_gal_iters)

nm_results[:combined] =
  combine_star_gal(nm_results[:star], nm_results[:galaxy])
bfgs_results[:combined] =
  combine_star_gal(bfgs_results[:star], bfgs_results[:galaxy])
print_params(nm_results[:combined], bfgs_results[:combined])
ElboDeriv.get_brightness(nm_results[:combined])
ElboDeriv.get_brightness(bfgs_results[:combined])

nm_results[:a] = fit_type(:a, nm_results[:combined], newton_fit_params)[1]
bfgs_results[:a] = fit_type(:a, bfgs_results[:combined], bfgs_fit_params)[1]
print_params(nm_results[:a], bfgs_results[:a])

# For some reason this sets a to be 0.5...
nm_results[:both], nm_both_iters, nm_both_v =
fit_type(:both, nm_results[:a], newton_fit_params)
bfgs_results[:both], nm_both_iters, nm_both_v =
  fit_type(:a, bfgs_results[:a], bfgs_fit_params)



######################
# Print results
print_params(mp_original, mp_optim, mp_bfgs)

ElboDeriv.get_brightness(mp_original)[1]
ElboDeriv.get_brightness(mp_optim)[1]
ElboDeriv.get_brightness(mp_bfgs)[1]


##########################
# Simpler tests

function verify_sample_star(vs, pos)
    @test vs[ids.a[2]] <= 0.011

    @test_approx_eq_eps vs[ids.u[1]] pos[1] 0.1
    @test_approx_eq_eps vs[ids.u[2]] pos[2] 0.1

    brightness_hat = vs[ids.r1[1]] * vs[ids.r2[1]]
    @test_approx_eq_eps brightness_hat / sample_star_fluxes[3] 1. 0.01

    true_colors = log(sample_star_fluxes[2:5] ./ sample_star_fluxes[1:4])
    for b in 1:4
        @test_approx_eq_eps vs[ids.c1[b, 1]] true_colors[b] 0.2
    end
end

function verify_sample_galaxy(vs, pos)
    @test vs[ids.a[2]] >= 0.98

    @test_approx_eq_eps vs[ids.u[1]] pos[1] 0.1
    @test_approx_eq_eps vs[ids.u[2]] pos[2] 0.1

    @test_approx_eq_eps vs[ids.e_axis] .7 0.05
    @test_approx_eq_eps vs[ids.e_dev] 0.1 0.08
    @test_approx_eq_eps vs[ids.e_scale] 4. 0.2

    phi_hat = vs[ids.e_angle]
    phi_hat -= floor(phi_hat / pi) * pi
    five_deg = 5 * pi/180
    @test_approx_eq_eps phi_hat pi/4 five_deg

    brightness_hat = vs[ids.r1[2]] * vs[ids.r2[2]]
    @test_approx_eq_eps brightness_hat / sample_galaxy_fluxes[3] 1. 0.01

    true_colors = log(sample_galaxy_fluxes[2:5] ./ sample_galaxy_fluxes[1:4])
    for b in 1:4
        @test_approx_eq_eps vs[ids.c1[b, 2]] true_colors[b] 0.2
    end
end

omitted_ids = [ids_free.k[:], ids_free.c2[:], ids_free.r2]
blob, mp_original, body = gen_sample_star_dataset();

mp_bfgs = deepcopy(mp_original);
OptimizeElbo.maximize_likelihood(blob, mp_bfgs, transform);
verify_sample_star(mp_bfgs.vp[1], [10.1, 12.2]);

mp = deepcopy(mp_original);
iter_count, max_f, max_x, ret =
    OptimizeElbo.maximize_f_newton(
      ElboDeriv.elbo, mp, tiled_blob, transform,
      omitted_ids=omitted_ids, verbose=false, max_iters=10);
print_params(mp, mp_bfgs)
verify_sample_star(mp.vp[1], [10.1, 12.2])











##########################
#########################
# Newton's method by hand, probably obsolete.

obj_wrap = OptimizeElbo.ObjectiveWrapperFunctions(
    mp -> ElboDeriv.elbo(blob, mp),
    deepcopy(mp_original), transform, kept_ids, omitted_ids);
# For minimization, which is required by the linesearch algorithm.
obj_wrap.state.scale = -1.0
x0 = transform.vp_to_vector(mp_original.vp, omitted_ids);
elbo_grad = zeros(Float64, length(x0));
elbo_hess = zeros(Float64, length(x0), length(x0));


function f_grad!(x, grad)
  grad[:] = obj_wrap.f_grad(x)
end
d = Optim.DifferentiableFunction(obj_wrap.f_value, f_grad!,
                                 obj_wrap.f_value_grad!);

f_vals = zeros(Float64, max_iters);
cumulative_iters = zeros(Int64, max_iters);
x_vals = [ zeros(Float64, length(x0)) for iter=1:max_iters ];
grads = [ zeros(Float64, length(x0)) for iter=1:max_iters ];
hesses = [ zeros(Float64, length(x0), length(x0)) for iter=1:max_iters ];

# warm start with BFGS
warm_start = false
if warm_start
    mp_start = deepcopy(mp_original)
    start_iter_count, start_f, x_start =
      OptimizeElbo.maximize_f(ElboDeriv.elbo, blob, mp_start,
                              Transform.free_transform,
                              omitted_ids=omitted_ids, ftol_abs=1);
    obj_wrap.state.f_evals = start_iter_count;
    x_new = deepcopy(x_start); # For quick restarts while debugging
    new_val = old_val = -start_f;
else
    x_new = transform.vp_to_vector(mp_original.vp, omitted_ids);
    obj_wrap.state.f_evals = 0
    new_val = old_val = obj_wrap.f_value(x_new);
end

for iter in 1:max_iters
    println("-------------------$iter")
    x_old = deepcopy(x_new);
    old_val = new_val;

    elbo_hess = obj_wrap.f_ad_hessian(x_new);
    hesses[iter] = elbo_hess
    hess_ev = eig(elbo_hess)[1]
    min_ev = minimum(hess_ev)
    max_ev = maximum(hess_ev)
    println("========= Eigenvalues: $(max_ev), $(min_ev)")
    if min_ev < 0
        println("========== Warning -- non-convex, $(min_ev)")
        elbo_hess += eye(length(x_new)) * abs(min_ev) * 2
        hess_ev = eig(elbo_hess)[1]
        min_ev = minimum(hess_ev)
        max_ev = maximum(hess_ev)
        println("========= New eigenvalues: $(max_ev), $(min_ev)")
    end
    # if abs(max_ev) / abs(min_ev) > 1e6
    #     println("Regularizing hessian")
    #     elbo_hess += eye(length(x_new)) * (abs(max_ev) / 1e6)
    #     hess_ev = eig(elbo_hess)[1]
    #     min_ev = minimum(hess_ev)
    #     max_ev = maximum(hess_ev)
    #     println("========= New eigenvalues: $(max_ev), $(min_ev)")
    # end
    f_val, gr_new = obj_wrap.f_value_grad(x_old);
    x_direction = -(elbo_hess \ gr_new)

    lsr = Optim.LineSearchResults(Float64); # Not used
    c = -1.; # Not used
    mayterminate = true; # Not used
    pre_linesearch_iters = obj_wrap.state.f_evals
    interpolating_linesearch!(d, x_old, x_direction,
                              x_new, gr_new,
                              lsr, c, mayterminate;
                              c1 = 1e-4,
                              c2 = 0.9,
                              rho = 2.0, verbose=false);
    new_val, gr_new = obj_wrap.f_value_grad(x_new)
    println("Spent $(obj_wrap.state.f_evals - pre_linesearch_iters) iterations ",
             "on linesearch for an extra $(f_val - new_val).")
    val_diff = new_val / old_val - 1
    f_vals[iter] = new_val;
    x_vals[iter] = deepcopy(x_new)
    grads[iter] = deepcopy(gr_new)
    cumulative_iters[iter] = obj_wrap.state.f_evals
    println(">>>>>>  Current value after $(obj_wrap.state.f_evals) evaluations: ",
            "$(new_val) (BFGS got $(-bfgs_v) in $(iter_count) iters)")
    mp_nm = deepcopy(mp_original);
    transform.vector_to_vp!(x_new, mp_nm.vp, omitted_ids);
    #println(ElboDeriv.get_brightness(mp_nm))
    println("\n\n")
end

# f_vals are negative because it's minimization
println("Newton objective - BFGS objective (higher is better)")
println("Cumulative fuction evaluation ratio:")
hcat(((-f_vals) - bfgs_v) / abs(bfgs_v), cumulative_iters ./ iter_count)

reduce(hcat, [ x_diff ./ x_vals[1] for x_diff in diff(x_vals) ])

mp_nm = deepcopy(mp_original);
transform.vector_to_vp!(x_new, mp_nm.vp, omitted_ids);
