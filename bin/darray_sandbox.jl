#!/usr/bin/env julia
using Celeste
using CelesteCluster

# Use nw workers.
nw = 5

CelesteCluster.create_workers(nw);

# Strangely the libraries cannot be loaded from within the module.
println("Loading libraries.")
@everywhere begin
  using Celeste
  using CelesteTypes
  using JLD
  import SampleData.dat_dir
  frame_jld_file = "initialzed_celeste_003900-6-0269.JLD"
  synthetic = false
end

println("Loading data.")
@everywhere begin
  if synthetic
    srand(1)
    S = 100
    blob, mp, body, tiled_blob =
      SampleData.gen_n_body_dataset(S, tile_width=10);
  else
    img_dict = JLD.load(joinpath(dat_dir, frame_jld_file));
    tiled_blob = img_dict["tiled_blob"];
    mp = img_dict["mp_all"];
  end
end;



CelesteCluster.initialize_cluster(frame_jld_file, nw);

# Define the elbo evaluation function.
# TODO: maybe this should be in ElboDeriv.
@everywhere eval_likelihood! =
  remotecall_fetch(1, () -> CelesteCluster.eval_likelihood!)


# Sanity check that the tiles were communicated successfully
@everywhere tilesum = sum([sum([ sum(t.pixels) for t in tiled_blob[b]]) for b=1:5 ])
remote_tilesums = [remotecall_fetch(w, () -> tilesum) for w in workers()];
@assert tilesum == sum(remote_tilesums)

# Sanity check that the mp is the same.
for w in workers()
  @assert mp.vp == remotecall_fetch(w, () -> mp.vp)
end

#######################################
# Evaluate the elbo.

# locally:
@time elbo_time = eval_likelihood()

# Evaluate the ELBO in parallel.  Most of the time is taken up on the workers.
@time begin
  @everywhereelse elbo_time = eval_likelihood();
  accum_workers = [ fetch(rr) for rr in accum_rr];
  accum_par = sum(accum_workers);
end;

# Sometimes this is faster indicating that you're processer bound in the
# parallel execution.
@runat 2 begin
  @time eval_likelihood()
end

# Make sure they match.
@assert abs(accum.v / accum_par.v - 1) < 1e-6
@assert maximum(abs((accum.d .+ 1e-8) ./ (accum_par.d .+ 1e-8) - 1)) < 1e-6

elbo_times = [ remotecall_fetch(w, () -> elbo_time) for w in workers() ]
elbo_time / maximum(elbo_times)
elbo_time / nw
num_sources = [ remotecall_fetch(w, () -> length(local_sources())) for w in workers() ]

elbo_times ./ num_sources
