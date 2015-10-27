#!/usr/bin/env julia
using Celeste
using PyPlot

include(joinpath(Pkg.dir("Celeste"), "src/CelesteCluster.jl"))
frame_jld_file = "initialzed_celeste_003900_6_0269.JLD"
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
    min_index = findfirst(cumsum(nonzero_pixels ./ sum(nonzero_pixels)) .>= 1e-2)
    threshold = nonzero_pixels[min_index]
    for h=1:H, w=1:W
      if s in mp.tile_sources[b][h, w] && pred_tiles[b][h, w] < threshold
        mp.tile_sources[b][h, w] = setdiff(mp.tile_sources[b][h, w], [s])
      end
    end
  end
end


function stitch_object_tiles(
    s::Int64, b::Int64, mp::ModelParams{Float64}, tiled_blob::TiledBlob)

  has_s = Bool[ s in mp.tile_sources[b][h, w] for h=1:H, w=1:W ];
  tiles_s = tiled_blob[b][has_s];
  h_range = Int[typemax(Int), typemin(Int)]
  w_range = Int[typemax(Int), typemin(Int)]
  for tile in tiles_s
    h_range[1] = min(minimum(tile.h_range), h_range[1])
    h_range[2] = max(maximum(tile.h_range), h_range[2])
    w_range[1] = min(minimum(tile.w_range), w_range[1])
    w_range[2] = max(maximum(tile.w_range), w_range[2])
  end

  image_s = fill(NaN, diff(h_range)[1] + 1, diff(w_range)[1] + 1);
  for tile in tiles_s
    image_s[tile.h_range - h_range[1] + 1, tile.w_range - w_range[1] + 1] =
      tile.pixels
  end
  image_s
end


mp_1 = deepcopy(mp);
mp_1.active_sources = [1];
trim_source_tiles!(1, mp_1);

[ sum([ 1 in tile for tile in mp.tile_sources[b]]) for b in 1:5]
[ sum([ 1 in tile for tile in mp_1.tile_sources[b]]) for b in 1:5]

for b=1:5
  image_s = stitch_object_tiles(1, b, mp_1, tiled_blob);
  matshow(image_s); colorbar(); title(b)
end


@time ElboDeriv.elbo(tiled_blob, mp_1);
transform = Transform.get_mp_transform(mp_1);
OptimizeElbo.maximize_elbo(tiled_blob, mp_1; verbose=true);





# ok
