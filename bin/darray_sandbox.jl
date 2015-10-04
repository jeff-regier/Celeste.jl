#!/usr/bin/env julia

# Like @everywhere but for remote workers.
macro everywhereelse(ex)
    quote
        @sync begin
            for w in workers()
                @async remotecall_fetch(w, ()->(eval(Main,$(Expr(:quote,ex))); nothing))
            end
        end
    end
end

# Start with julia -p <n_workers.
# Make sure there are at least five workers:
nw = 5
if length(workers()) < 5
  addprocs(nw - length(workers()))
end

# http://julia.readthedocs.org/en/release-0.3/manual/parallel-computing/
@everywhere using Celeste
@everywhere using CelesteTypes

import Synthetic
import ModelInit
import SampleData

const dat_dir = joinpath(Pkg.dir("Celeste"), "dat")
println("Running with ", length(workers()), " processors.")

@everywhere begin
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
end

srand(1)
println("Loading data.")

S = 100
blob, mp, body, tiled_blob =
  SampleData.gen_n_body_dataset(S, tile_width=10);

NumType = Float64
tiles = tiled_blob[3];
b = 3;
accum = zero_sensitive_float(CanonicalParams, NumType, mp.S);

star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(mp, b);
sbs = ElboDeriv.SourceBrightness{NumType}[
  ElboDeriv.SourceBrightness(mp.vp[s]) for s in 1:mp.S];

# Divide up the work
nw = length(workers())
col_cuts = iround(linspace(1, size(tiles)[2] + 1, nw + 1))
col_ranges = map(i -> col_cuts[i]:col_cuts[i + 1] - 1, 1:nw)

@everywhereelse tiles_rr = RemoteRef(1)
tiles_rr = [ remotecall_fetch(w, () -> tiles_rr) for w in workers() ]
for iw=1:nw
  put!(tiles_rr[iw], tiles[:,col_ranges[iw]])
end

mp_rr = RemoteRef(1)
sbs_rr = RemoteRef(1)
gal_mcs_rr = RemoteRef(1)
star_mcs_rr = RemoteRef(1)

put!(mp_rr, mp);
put!(sbs_rr, sbs);
put!(gal_mcs_rr, gal_mcs);
put!(star_mcs_rr, star_mcs);


@everywhereelse begin
  mp_rr = remotecall_fetch(1, () -> mp_rr);
  mp = fetch(mp_rr);

  sbs_rr = remotecall_fetch(1, () -> sbs_rr);
  sbs = fetch(sbs_rr);

  star_mcs_rr = remotecall_fetch(1, () -> star_mcs_rr);
  star_mcs = fetch(star_mcs_rr);

  gal_mcs_rr = remotecall_fetch(1, () -> gal_mcs_rr);
  gal_mcs = fetch(gal_mcs_rr);

  tiles = fetch(tiles_rr);
  accum = zero_sensitive_float(CanonicalParams, Float64, mp.S)
  accum_rr = RemoteRef(myid())
  put!(accum_rr, accum)
end

accum_rr = [remotecall_fetch(w, () -> accum_rr) for w in workers()];

# Evaluate the ELBO
tic()
@everywhereelse begin
  #println(myid(), " is starting")
  for tile in tiles
    #print(myid())
    tile_sources = mp.tile_sources[3][tile.hh, tile.ww]
    tile_likelihood!(
      tile, tile_sources, mp, sbs, star_mcs, gal_mcs, accum)
  end
  #println(myid(), " is done")
end
accum_par = sum([ fetch(rr) for rr in accum_rr]);
toc()

# Do it in serial
accum = zero_sensitive_float(CanonicalParams, Float64, mp.S);
tic()
for tile in tiles
  tile_sources = mp.tile_sources[3][tile.hh, tile.ww]
  tile_likelihood!(
    tile, tile_sources, mp, sbs, star_mcs, gal_mcs, accum)
end
toc()
abs(accum.v - accum_par.v)
maximum(abs(accum.d - accum_par.d))
