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

# Sanity check
@everywhereelse tilesum = sum([ sum(t.pixels) for t in tiles])
tilesum = sum([ sum(t.pixels) for t in tiles])
@assert tilesum == sum([ remotecall_fetch(w, () -> tilesum) for w in workers() ])

# Likelihood
@everywhere begin
  function eval_likelihood()
    println(myid(), " is starting.")
    elbo_time = time()
    global accum
    clear!(accum)
    for tile in tiles
      tile_sources = mp.tile_sources[3][tile.hh, tile.ww]
      tile_likelihood!(
        tile, tile_sources, mp, sbs, star_mcs, gal_mcs, accum)
    end
    elbo_time = time() - elbo_time
    println(myid(), " is done in $(elbo_time)s.")
  end
end

# The value of accum in the remote ref is not getting updated.
@everywhere begin
  function set_accum(x::Float64)
    global accum
    accum.v = x
  end
end

@everywhereelse set_accum(7.0)
[ fetch(rr).v for rr in accum_rr] # Doesn't work
[ remotecall_fetch(w, () -> accum.v) for w in workers() ] # works

# Try setting the RemoteRefs on the remote ones
accum_rr = [ RemoteRef(w) for w in workers() ]
@everywhereelse begin
  # The worker ids are not in the same order on every process
  worker_ids = remotecall_fetch(1, workers)
  proc_id = find(worker_ids .== myid())
  @assert length(proc_id) == 1
  accum_rr = remotecall_fetch(1, i -> accum_rr[i], proc_id[1])
  put!(accum_rr, accum)
end

# Now this works
@everywhereelse set_accum(9.0)
[ fetch(rr).v for rr in accum_rr] # Doesn't work
[ remotecall_fetch(w, () -> accum.v) for w in workers() ] # works



# Evaluate the ELBO in parallel
@time begin
  @everywhereelse eval_likelihood();
  accum_par = sum([ fetch(rr) for rr in accum_rr]);
end

# Do it in serial
@time eval_likelihood()
@assert abs(accum.v - accum_par.v) < 1e-6
@assert maximum(abs(accum.d - accum_par.d)) < 1e-6
