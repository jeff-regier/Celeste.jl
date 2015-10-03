#!/usr/bin/env julia

# Start with julia -p <n_workers.
# Make sure there are at least three workers:
if length(workers()) < 3
  addprocs(3 - length(workers()))
end

# http://julia.readthedocs.org/en/release-0.3/manual/parallel-computing/
@everywhere using Celeste
@everywhere using CelesteTypes

import Synthetic
import ModelInit
import SampleData

const dat_dir = joinpath(Pkg.dir("Celeste"), "dat")
println("Running with ", length(workers()), " processors.")

using ElboDeriv.tile_likelihood!
using ElboDeriv.SourceBrightness
using ElboDeriv.BvnComponent
using ElboDeriv.GalaxyCacheComponent

function elbo_likelihood!{NumType <: Number}(
  tile::ImageTile, mp::ModelParams{NumType},
  sbs::Vector{SourceBrightness{NumType}},
  star_mcs::Array{BvnComponent{NumType}, 2},
  gal_mcs::Array{GalaxyCacheComponent{NumType}, 4},
  accum::SensitiveFloat{CanonicalParams, NumType})
    tile_sources = mp.tile_sources[b][tile.hh, tile.ww]
    tile_likelihood!(
      tile, tile_sources, mp, sbs, star_mcs, gal_mcs, accum)
end


srand(1)
println("Loading data.")

S = 10
blob, mp, body, tiled_blob =
  SampleData.gen_n_body_dataset(S, tile_width=10);

NumType = Float64
tiles = tiled_blob[3];
b = 3;
accum = zero_sensitive_float(CanonicalParams, NumType, mp.S);

star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(mp, b);
sbs = ElboDeriv.SourceBrightness{NumType}[
  ElboDeriv.SourceBrightness(mp.vp[s]) for s in 1:mp.S];


#dtiles = distribute(tiles);

function init_dtiles()
  # Same as distribute but with more control.

  # Not sure why, but I can't run this in the global space,
  # since it says <owner> is not found.
  owner = myid()
  rr = RemoteRef();
  put!(rr, tiles);
  dtiles = DArray(size(tiles), workers(), [1, length(workers())]) do I
      remotecall_fetch(owner, ()->fetch(rr)[I...])
  end;
  dtiles
end
dtiles = init_dtiles();
for worker in workers()
  println(remotecall_fetch(worker, localindexes, dtiles))
end

remotecall_fetch(2, myid)
@spawnat workers()[1] remotecall_fetch(1, myid)

@spawnat workers()[1] foo = remotecall_fetch(1, myid)


mp_rr = RemoteRef(1)
put!(mp_rr, mp);
numtype_rr = RemoteRef(1)
put!(numtype_rr, NumType)


# This works
remotecall_fetch(2, fetch, numtype_rr)

# This does not
@everywhere NumType = fetch(numtype_rr)
@everywhere println(fetch(numtype_rr))

rr_2 = remotecall_fetch(2, RemoteRef, 2)
put!(rr_2, 10)
@everywhere println(fetch(rr_2))



@everywhere function init_sf(I)
  NumType = fetch(numtype_rr)
  mp = fetch(mp_rr)
  fill(zero_sensitive_float(CanonicalParams, NumType, mp.S),
     map(length, I))
end

remotecall_fetch(2, init_sf, 1:4)

daccum = DArray(init_sf, size(dtiles), procs(dtiles), [size(dtiles.chunks)...]);

@everywhere @assert localindexes(dtiles) == localindexes(daccum)

@everywhere function get_tile_sf(hh::Int64, ww::Int64)
  tile_accum = zero_sensitive_float(CanonicalParams, NumType, mp.S)
  tile_sources = mp.tile_sources[b][tile.hh, tile.ww]
  tile_likelihood!(
    tile, tile_sources, mp, sbs, star_mcs, gal_mcs, tile_accum)
  tile_accum
end




accum_par = @parallel (+) for tile in tiles
  get_tile_sf(tile)
end

# TODO: why doesn't @parallel update something in place?
accum.v += accum_par.v
accum.d += accum_par.d
accum.h += accum_par.h

else
  for tile in tiles
    tile_sources = mp.tile_sources[b][tile.hh, tile.ww]
    tile_likelihood!(
      tile, tile_sources, mp, sbs, star_mcs, gal_mcs, accum)
  end
end
