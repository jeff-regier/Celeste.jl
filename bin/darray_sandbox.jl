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

# This is like @everywhere but only runs on a particular process.
macro runat(p, ex)
  quote
    remotecall_fetch($p, ()->(eval(Main,$(Expr(:quote,ex))); nothing))
  end
end

# Start with julia -p <n_workers.
# Make sure there are at least nw workers:
nw = 3
if length(workers()) < nw
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
original_tiles = tiles;
accum = zero_sensitive_float(CanonicalParams, NumType, mp.S);

# Divide up the work
nw = length(workers())

distributed_tiles = Array(Array{ImageTile}, nw);
for w=1:nw
  distributed_tiles[w] = ImageTile[];
end

for b=1:5
  col_cuts = iround(linspace(1, size(tiled_blob[b])[2] + 1, nw + 1))
  col_ranges = map(i -> col_cuts[i]:col_cuts[i + 1] - 1, 1:nw)
  for w=1:nw
    append!(distributed_tiles[w], tiled_blob[b][:, col_ranges[w]][:])
  end
end

# All the information that needs to be communicated to the tile processors.
type ParamState{NumType <: Number}
  mp::ModelParams{NumType}
  star_mcs_vec::Vector{Array{BvnComponent{NumType},2}}
  gal_mcs_vec::Vector{Array{GalaxyCacheComponent{NumType},4}}
  sbs_vec::Vector{Vector{SourceBrightness{NumType}}}
end

ParamState{NumType <: Number}(mp::ModelParams{NumType}) = begin
  num_bands = size(mp.patches)[2]
  star_mcs_vec = Array(Array{BvnComponent{NumType},2}, num_bands)
  gal_mcs_vec = Array(Array{GalaxyCacheComponent{NumType},4}, num_bands)
  sbs_vec = Array(Vector{SourceBrightness{NumType}}, num_bands)
  ParamState(mp, star_mcs_vec, gal_mcs_vec, sbs_vec)
end

NumType = Float64
param_state = ParamState(mp);
for b=1:5
  star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(mp, b);
  sbs = ElboDeriv.SourceBrightness{NumType}[
    ElboDeriv.SourceBrightness(mp.vp[s]) for s in 1:mp.S];

  param_state.star_mcs_vec[b] = star_mcs
  param_state.gal_mcs_vec[b] = gal_mcs
  param_state.sbs_vec[b] = sbs
end




# Since this is used to index into data structures, it must
# be the same on every process.
worker_ids = workers()
@everywhereelse worker_ids = remotecall_fetch(1, () -> worker_ids)

# Copy the tiles.
@everywhereelse tiles_rr = RemoteRef(1)
tiles_rr = [ remotecall_fetch(w, () -> tiles_rr) for w in workers() ]
for iw=1:nw
  put!(tiles_rr[iw], tiles[:,col_ranges[iw]])
end

@everywhereelse mp_rr = RemoteRef(1)
mp_rr = [ remotecall_fetch(w, () -> mp_rr) for w in workers() ]
for rr in mp_rr
  put!(rr, mp)
end
@everywhereelse mp = fetch(mp_rr)

@everywhereelse sbs_rr = RemoteRef(1)
sbs_rr = [ remotecall_fetch(w, () -> sbs_rr) for w in workers() ]
for rr in sbs_rr
  put!(rr, sbs)
end
@everywhereelse sbs = fetch(sbs_rr)

@everywhereelse star_mcs_rr = RemoteRef(1)
star_mcs_rr = [ remotecall_fetch(w, () -> star_mcs_rr) for w in workers() ]
for rr in star_mcs_rr
  put!(rr, star_mcs)
end
@everywhereelse star_mcs = fetch(star_mcs_rr)

@everywhereelse gal_mcs_rr = RemoteRef(1)
gal_mcs_rr = [ remotecall_fetch(w, () -> gal_mcs_rr) for w in workers() ]
for rr in gal_mcs_rr
  put!(rr, gal_mcs)
end
@everywhereelse gal_mcs = fetch(gal_mcs_rr)

# Note that this updates automatically:
# mp.vp[1][1] = 5.0
# @everywhereelse mp = fetch(mp_rr)
# @everywhereelse println(mp.vp[1][1])

# Set up for accum to be communicated back to process 1
accum_rr = [ RemoteRef(w) for w in workers() ]
@everywhereelse begin
  tiles = fetch(tiles_rr);
  accum = zero_sensitive_float(CanonicalParams, Float64, mp.S)
  accum_rr = RemoteRef(myid())
  proc_id = find(worker_ids .== myid())
  @assert length(proc_id) == 1
  accum_rr = remotecall_fetch(1, i -> accum_rr[i], proc_id[1])
  put!(accum_rr, accum)
end

# Sanity check that the tiles were communicated successfully
@everywhereelse tilesum = sum([ sum(t.pixels) for t in tiles])
tilesum = sum([ sum(t.pixels) for t in tiles])
@assert tilesum == sum([ remotecall_fetch(w, () -> tilesum) for w in workers() ])

# Sanity check that the mp is the same.
for w in workers()
  @assert mp.vp == remotecall_fetch(w, () -> mp.vp)
end

# Define the likelihood function
@everywhere begin
  function eval_likelihood()
    println(myid(), " is starting.")
    elbo_time = time()
    global accum
    clear!(accum)
    for tile in tiles
      tile_sources = mp.tile_sources[3][tile.hh, tile.ww]
      tile_likelihood!(
        tile, tile_sources, mp, sbs, star_mcs, gal_mcs, accum);
    end
    elbo_time = time() - elbo_time
    println(myid(), " is done in $(elbo_time)s.")
  end
end


# Evaluate the ELBO in parallel.  Most of the time is taken up on the workers.
@time begin
  @everywhereelse eval_likelihood();
  accum_workers = [ fetch(rr) for rr in accum_rr];
  accum_par = sum(accum_workers);
end;

tiles = original_tiles;
@time eval_likelihood()

# Make sure they match.
@assert abs(accum.v / accum_par.v - 1) < 1e-6
@assert maximum(abs((accum.d .+ 1e-8) ./ (accum_par.d .+ 1e-8) - 1)) < 1e-6

# Note that each iteration takes less time on process 1.  Why?
for id=1:nw
  tiles = original_tiles[:, col_ranges[id]]
  eval_likelihood()
end


# Profiling.  The two look similar.
tiles = original_tiles[:, col_ranges[2]];
@runat 2 begin
  Profile.init(10^8, 0.001)
  @profile eval_likelihood()
  data, ldict = Profile.retrieve()
end
data2 = remotecall_fetch(2, () -> data);
ldict2 = remotecall_fetch(2, () -> ldict);

Profile.init(10^8, 0.001)
@profile eval_likelihood()
data, ldict = Profile.retrieve();

Profile.print(data2, ldict2)
println("-----------------")
Profile.print(data, ldict)

# The same amount of memory is allocated.
@runat 2 begin
  @time eval_likelihood()
end

@time eval_likelihood()
