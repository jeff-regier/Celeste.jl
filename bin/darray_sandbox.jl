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
nw = 10
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

  function eval_likelihood()
    println(myid(), " is starting.")
    elbo_time = time()
    global accum
    clear!(accum)
    for b in 1:5
      println("Proc $(myid()) starting band $b")
      sbs = param_state.sbs_vec[b]
      star_mcs = param_state.star_mcs_vec[b]
      gal_mcs = param_state.gal_mcs_vec[b]
      for tile in tiled_blob[b][:]
        tile_sources = mp.tile_sources[b][tile.hh, tile.ww]
        tile_likelihood!(
          tile, tile_sources, mp,
          sbs, star_mcs, gal_mcs,
          accum);
      end
    end
    elbo_time = time() - elbo_time
    println(myid(), " is done in $(elbo_time)s.")
  end

end

srand(1)
println("Loading data.")

S = 100
blob, mp, body, tiled_blob =
  SampleData.gen_n_body_dataset(S, tile_width=10);

NumType = Float64
accum = zero_sensitive_float(CanonicalParams, NumType, mp.S);

#######################################
# Divide up the tiles
nw = length(workers())

distributed_tiled_blobs = Array(TiledBlob, nw);
for w=1:nw
  distributed_tiled_blobs[w] = Array(TiledImage, 5);
end

for b=1:5
  col_cuts = iround(linspace(1, size(tiled_blob[b])[2] + 1, nw + 1))
  col_ranges = map(i -> col_cuts[i]:col_cuts[i + 1] - 1, 1:nw)
  for w=1:nw
    distributed_tiled_blobs[w][b] = tiled_blob[b][:, col_ranges[w]]
  end
end

# Copy the tiles.
@everywhereelse tiles_rr = RemoteRef(1)
tiles_rr = [ remotecall_fetch(w, () -> tiles_rr) for w in workers() ]
for iw=1:nw
  put!(tiles_rr[iw], distributed_tiled_blobs[iw])
end

@everywhereelse tiled_blob = fetch(tiles_rr);

# Sanity check that the tiles were communicated successfully
@everywhereelse tilesum = sum([ sum(t.pixels) for t in tiles])
tilesum = sum([sum([ sum(t.pixels) for t in tiled_blob[b]]) for b=1:5 ])
@assert tilesum == sum([remotecall_fetch(w, () -> tilesum) for w in workers()])


#######################################
# All the information that needs to be communicated to the tile processors.
@everywhere type ParamState{NumType <: Number}
  vp::VariationalParams{NumType}
  star_mcs_vec::Vector{Array{BvnComponent{NumType},2}}
  gal_mcs_vec::Vector{Array{GalaxyCacheComponent{NumType},4}}
  sbs_vec::Vector{Vector{SourceBrightness{NumType}}}
end

ParamState{NumType <: Number}(mp::ModelParams{NumType}) = begin
  num_bands = size(mp.patches)[2]
  star_mcs_vec = Array(Array{BvnComponent{NumType},2}, num_bands)
  gal_mcs_vec = Array(Array{GalaxyCacheComponent{NumType},4}, num_bands)
  sbs_vec = Array(Vector{SourceBrightness{NumType}}, num_bands)
  ParamState(mp.vp, star_mcs_vec, gal_mcs_vec, sbs_vec)
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

# Copy the model params.
@everywhereelse mp = remotecall_fetch(1, () -> mp);

# Set up sockets for the ParamState
@everywhereelse param_state_rr = RemoteRef(1)
param_state_rr = [remotecall_fetch(w, () -> param_state_rr) for w in workers()]
for rr in param_state_rr
  put!(rr, param_state)
end
@everywhereelse param_state = fetch(param_state_rr)

# Set up for accum to be communicated back to process 1

# Since this is used to index into accum_rr, it must
# be the same on every process.
worker_ids = workers()
@everywhereelse worker_ids = remotecall_fetch(1, () -> worker_ids)

accum_rr = [ RemoteRef(w) for w in workers() ]
@everywhereelse begin
  accum = zero_sensitive_float(CanonicalParams, Float64, mp.S)
  accum_rr = RemoteRef(myid())
  proc_id = find(worker_ids .== myid())
  @assert length(proc_id) == 1
  accum_rr = remotecall_fetch(1, i -> accum_rr[i], proc_id[1])
  put!(accum_rr, accum)
end

# Sanity check that the mp is the same.
for w in workers()
  @assert mp.vp == remotecall_fetch(w, () -> mp.vp)
end


#######################################
# Define the likelihood function

# Evaluate the elbo locally.
@time eval_likelihood()

# Evaluate the ELBO in parallel.  Most of the time is taken up on the workers.
@time begin
  @everywhereelse eval_likelihood();
  accum_workers = [ fetch(rr) for rr in accum_rr];
  accum_par = sum(accum_workers);
end;

# Make sure they match.
@assert abs(accum.v / accum_par.v - 1) < 1e-6
@assert maximum(abs((accum.d .+ 1e-8) ./ (accum_par.d .+ 1e-8) - 1)) < 1e-6


######################################
# Profiling.
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
