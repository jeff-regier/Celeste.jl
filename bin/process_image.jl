#!/usr/bin/env julia
println("Running process_image.jl")
println(versioninfo())

using Celeste
#using PyPlot
using ArgParse
using Compat

# Command line arguments
s = ArgParseSettings()
@add_arg_table s begin
    "--sources"
        help = "An array of source ids (as a parseable expression, e.g. [1,2,3])"
        arg_type = ASCIIString
        default = "[1]"
    "--image_file"
        help = "The image JLD file"
        default = "initialzed_celeste_003900_6_0269_5px.JLD"
end

parsed_args = parse_args(s)
eval(parse(string("sources = Int64", parsed_args["sources"])))
@assert length(sources) > 0

frame_jld_file = parsed_args["image_file"]

include(joinpath(Pkg.dir("Celeste"), "src/CelesteCluster.jl"))
#frame_jld_file = "initialzed_celeste_003900_6_0269_20px.JLD"
#frame_jld_file = "initialzed_celeste_003900_6_0269_10px_nopsf.JLD"
#frame_jld_file = "initialzed_celeste_003900_6_0269_5px.JLD"
S = 20
synthetic = false

println("Loading data with sources = $(sources).")

if synthetic
  srand(1)
  blob, original_mp, body, tiled_blob =
    SampleData.gen_n_body_dataset(S, tile_width=10);
else
  img_dict = JLD.load(joinpath(dat_dir, frame_jld_file));
  tiled_blob = img_dict["tiled_blob"];
  original_mp = img_dict["mp_all"];
  cat_df = img_dict["cat_df"];
end;
mp = deepcopy(original_mp);

# Validate sources inputs
@assert maximum(sources) <= mp.S
@assert minimum(sources) >= 1
source = unique(sources)

omitted_ids = Int64[]

analysis_name = "iters_20"
max_iters = 20
for s in sources
  println("Processing source $s.")

  mp_s = deepcopy(mp);
  mp_s.active_sources = [s]

  # Some sanity checks and improved initialization.
  objid = mp.objids[s]
  cat_row = cat_df[:objid] .== objid;
  is_star = cat_df[cat_row, :is_star][1]
  mp_s.vp[s][ids.a[1]] = is_star ? 0.8: 0.2
  mp_s.vp[s][ids.a[2]] = 1.0 - mp_s.vp[s][ids.a][1]
  if cat_df[cat_row, :psfflux_r][1] < 10
    continue
  end

  transform = Transform.get_mp_transform(mp_s);
  trim_source_tiles!(s, mp_s);

  f = ElboDeriv.elbo
  elbo_time = time()
  iter_count, max_f, max_x, result =
    OptimizeElbo.maximize_f(ElboDeriv.elbo, tiled_blob, mp_s;
                            verbose=true, max_iters=max_iters);
  elbo_time = time() - elbo_time

  JLD.save("$dat_dir/elbo_fit_$(analysis_name)_s$(s)_$(time()).JLD",
           @compat(Dict("vp[s]" => mp_s.vp[s],
                        "s" => s,
                        "result" => result,
                        "elbo_time" => elbo_time,
                        "frame_jld_file" => frame_jld_file)));
end
println("All done!")


if false
  # Stuff for interactive exploration
  [ sum([ s in tile for tile in mp.tile_sources[b]]) for b in 1:5]
  [ sum([ s in tile for tile in mp_s.tile_sources[b]]) for b in 1:5]

  obj_cols = [:objid, :is_star, :is_gal, :psfflux_r, :compflux_r, :ra, :dec];
  sort(cat_df[cat_df[:is_gal] .== true, obj_cols],
      cols=:compflux_r, rev=true)
  sort(cat_df[cat_df[:is_gal] .== false, obj_cols],
      cols=:psfflux_r, rev=true)

  if false
    for b=1:5
      image_s = stitch_object_tiles(s, b, mp_s, tiled_blob);
      matshow(image_s); colorbar(); title(b)
    end
  end

  cat_df[cat_df[:objid] .== mp.objids[s], obj_cols]

  pred_images = Array(Matrix{Float64}, 5);
  cat_images = Array(Matrix{Float64}, 5);
  orig_images = Array(Matrix{Float64}, 5);

  mp_cat = deepcopy(mp_s);
  mp_cat.vp[s] = mp.vp[s];
  for b=1:5
    println("Band $b.")
    pred_images[b] = stitch_object_tiles(s, b, mp_s, tiled_blob; predicted=true);
    cat_images[b] = stitch_object_tiles(s, b, mp_cat, tiled_blob; predicted=true);
    orig_images[b] = stitch_object_tiles(s, b, mp_s, tiled_blob; predicted=false);
  end

  [ sum((pred_images[b] .- orig_images[b]).^2) for b=1:5]
  [ sum((cat_images[b] .- orig_images[b]).^2) for b=1:5]
  [ sum((pred_images[b] .- cat_images[b]).^2) for b=1:5]



  b = 2
  plot(orig_images[b], pred_images[b], "r.")
  plot(orig_images[b], cat_images[b], "b.")
  plot(40e3, 40e3, "ko")

  for b=1:5
    figure()
    subplot(1,4,1)
    imshow(pred_images[b]); colorbar(); title("Predicted $b")
    subplot(1,4,2)
    imshow(pred_images[b]); colorbar(); title("Catalog $b")
    subplot(1,4,3)
    imshow(orig_images[b]); colorbar(); title("Original $b")
    subplot(1,4,4)
    imshow(orig_images[b] - pred_images[b]); colorbar(); title("Difference $b")
  end
end


# ok
