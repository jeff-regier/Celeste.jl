#!/usr/bin/env julia
using Celeste
using PyPlot

include(joinpath(Pkg.dir("Celeste"), "src/CelesteCluster.jl"))
#frame_jld_file = "initialzed_celeste_003900_6_0269_20px.JLD"
frame_jld_file = "initialzed_celeste_003900_6_0269_10px_nopsf.JLD"
S = 20
synthetic = false

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


# Trim the tile_sources for source s.
function trim_source_tiles!(s::Int64, mp::ModelParams{Float64})
  pred_tiles = Array(Array{Float64}, 5);
  for b = 1:5
    H, W = size(tiled_blob[b])
    @assert size(mp.tile_sources[b]) == size(tiled_blob[b])
    pred_tiles[b] = Array(Float64, H, W)
    for h=1:H, w=1:W
      if s in mp.tile_sources[b][h, w]
        tile = tiled_blob[b][h, w]
        pred_tiles[b][h, w] =
          sum(ElboDeriv.tile_predicted_image(tile, mp, include_epsilon=false))
      else
        pred_tiles[b][h, w] = 0.0
      end
    end
    nonzero_pixels = sort(pred_tiles[b][pred_tiles[b] .> 0])
    println("Source $b sorted tiles: $(nonzero_pixels)")
    min_index = findfirst(cumsum(nonzero_pixels ./ sum(nonzero_pixels)) .>= 5e-2)
    threshold = nonzero_pixels[min_index]
    for h=1:H, w=1:W
      if s in mp.tile_sources[b][h, w] && pred_tiles[b][h, w] < threshold
        mp.tile_sources[b][h, w] = setdiff(mp.tile_sources[b][h, w], [s])
      end
    end
  end
end


function stitch_object_tiles(
    s::Int64, b::Int64, mp::ModelParams{Float64}, tiled_blob::TiledBlob;
    predicted::Bool=false)

  H, W = size(tiled_blob[b])
  has_s = Bool[ s in mp.tile_sources[b][h, w] for h=1:H, w=1:W ];
  tiles_s = tiled_blob[b][has_s];
  h_range = Int[typemax(Int), typemin(Int)]
  w_range = Int[typemax(Int), typemin(Int)]
  print("Stitching...")
  for tile in tiles_s
    print(".")
    h_range[1] = min(minimum(tile.h_range), h_range[1])
    h_range[2] = max(maximum(tile.h_range), h_range[2])
    w_range[1] = min(minimum(tile.w_range), w_range[1])
    w_range[2] = max(maximum(tile.w_range), w_range[2])
  end
  println("Done.")

  image_s = fill(NaN, diff(h_range)[1] + 1, diff(w_range)[1] + 1);
  for tile in tiles_s
    image_s[tile.h_range - h_range[1] + 1, tile.w_range - w_range[1] + 1] =
      predicted ?
      ElboDeriv.tile_predicted_image(tile, mp, include_epsilon=false):
      tile.pixels
  end
  image_s
end

obj_cols = [:objid, :is_star, :is_gal, :psfflux_r, :compflux_r, :ra, :dec];
sort(cat_df[cat_df[:is_gal] .== true, obj_cols],
    cols=:compflux_r, rev=true)
sort(cat_df[cat_df[:is_gal] .== false, obj_cols],
    cols=:psfflux_r, rev=true)

f = ElboDeriv.elbo;
omitted_ids = Int64[]
mp.active_sources = [s]

objid = "1237662226208063551"
s = find(mp.objids .== objid)[1]

mp_s = deepcopy(mp);
mp_s.active_sources = [s];
trim_source_tiles!(s, mp_s);

[ sum([ s in tile for tile in mp.tile_sources[b]]) for b in 1:5]
[ sum([ s in tile for tile in mp_s.tile_sources[b]]) for b in 1:5]

if false
  for b=1:5
    image_s = stitch_object_tiles(s, b, mp_s, tiled_blob);
    matshow(image_s); colorbar(); title(b)
  end
end

@time ElboDeriv.elbo(tiled_blob, mp_s);
transform = Transform.get_mp_transform(mp_s);

elbo_time = time()
OptimizeElbo.maximize_elbo(tiled_blob, mp_s; verbose=true);
elbo_time = time() - elbo_time

# The blob cannot be saved because it has a pointer to a C++ object.
JLD.save("$dat_dir/elbo_fit_s$(s)_$(time).JLD",
         Dict("vp[s]" => mp_s.vp[s],
              "s" => s,
              "elbo_time" => elbo_time,
              "frame_jld_file" => frame_jld_file));

cat_df[cat_df[:objid] .== mp.objids[s], obj_cols]

pred_images = Array(Matrix{Float64}, 5);
cat_images = Array(Matrix{Float64}, 5);
orig_images = Array(Matrix{Float64}, 5);

for b=1:5
  println("Band $b.")
  pred_images[b] = stitch_object_tiles(s, b, mp_s, tiled_blob; predicted=true);
  cat_images[b] = stitch_object_tiles(s, b, mp, tiled_blob; predicted=true);
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



# ok
