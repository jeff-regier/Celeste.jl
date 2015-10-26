#!/usr/bin/env julia
using Celeste
using PyPlot

include(joinpath(Pkg.dir("Celeste"), "src/CelesteCluster.jl"))
frame_jld_file = "initialzed_celeste_003900-6-0269.JLD"
S = 20
synthetic = true

load_cluster_data();

mp_1 = deepcopy(mp);
mp_1.active_sources = [1];

# Get predicted images.
s = 1;
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
end



transform = Transform.get_mp_transform(mp_1);
OptimizeElbo.maximize_elbo(tiled_blob, mp_1; verbose=true);
