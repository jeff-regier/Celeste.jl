#!/usr/bin/env julia

# Test the Celeste distributed over several workers.  This is
# slow due to having to load Celeste everywhere, maybe it will
# be faster with julia 0.4.

using Base.Test

# Use nw workers.
nw = 2
println("Adding workers.")
for worker in workers()
  if worker != myid()
    rmprocs(worker)
  end
end
addprocs(nw)
@assert(length(workers()) == nw)

println("Loading libraries.")
@everywhere begin
  using Celeste
  include(joinpath(Pkg.dir("Celeste"), "src/CelesteCluster.jl"))
  synthetic = true
  S = 10
end

load_cluster_data();
initialize_cluster();

# Sanity check that the tiles were communicated successfully
@everywhere begin
  tilesum =
    sum([sum([ sum(t.pixels) for t in tiled_blob[b]]) for b=1:5 ])
end
remote_tilesums =
  [remotecall_fetch(w, () -> tilesum) for w in workers()];
@test tilesum == sum(remote_tilesums)

# Sanity check that the mp is the same.
for w in workers()
  @test mp.vp == remotecall_fetch(w, () -> mp.vp)
end

#######################################
# Evaluate the elbo.

# locally:
accum = zero_sensitive_float(CanonicalParams, Float64, mp.S);
eval_worker_likelihood()

# Evaluate the ELBO in parallel.
@everywhereelse eval_worker_likelihood();
accum_workers = [ fetch(rr) for rr in accum_rr];
accum_par = sum(accum_workers);

# Make sure they match.
@test abs(accum.v / accum_par.v - 1) < 1e-6
@test maximum(abs((accum.d .+ 1e-8) ./ (accum_par.d .+ 1e-8) - 1)) < 1e-6
