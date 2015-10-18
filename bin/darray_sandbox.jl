#!/usr/bin/env julia

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

#create_workers(nw);

# Strangely the libraries cannot be loaded from within the module.
println("Loading libraries.")
@everywhere begin
  include(joinpath(Pkg.dir("Celeste"), "src/CelesteCluster.jl"))
  frame_jld_file = "initialzed_celeste_003900-6-0269.JLD"
  synthetic = true
end

load_cluster_data();
initialize_cluster();

# Define the elbo evaluation function.
# TODO: maybe this should be in ElboDeriv.
# @everywhere eval_worker_likelihood! =
#   remotecall_fetch(1, () -> CelesteCluster.eval_worker_likelihood!)

# Sanity check that the tiles were communicated successfully
@everywhere tilesum = sum([sum([ sum(t.pixels) for t in tiled_blob[b]]) for b=1:5 ])
remote_tilesums = [remotecall_fetch(w, () -> tilesum) for w in workers()];
@assert tilesum == sum(remote_tilesums)

# Sanity check that the mp is the same.
for w in workers()
  @assert mp.vp == remotecall_fetch(w, () -> mp.vp)
end

#######################################
# Evaluate the elbo.  Do it twice to avoid compile time.

# locally:
accum = zero_sensitive_float(CanonicalParams, Float64, mp.S);
@time elbo_time = eval_worker_likelihood()
@time elbo_time = eval_worker_likelihood()

# Evaluate the ELBO in parallel.  Most of the time is taken up on the workers.
@time begin
  @everywhereelse elbo_time = eval_worker_likelihood();
  accum_workers = [ fetch(rr) for rr in accum_rr];
  accum_par = sum(accum_workers);
end;

@time begin
  @everywhereelse elbo_time = eval_worker_likelihood();
  accum_workers = [ fetch(rr) for rr in accum_rr];
  accum_par = sum(accum_workers);
end;

# # Sometimes this is faster indicating that you're processer bound in the
# # parallel execution.
# @runat 2 begin
#   @time eval_worker_likelihood()
# end

# Make sure they match.
@assert abs(accum.v / accum_par.v - 1) < 1e-6
@assert maximum(abs((accum.d .+ 1e-8) ./ (accum_par.d .+ 1e-8) - 1)) < 1e-6

elbo_times = [ remotecall_fetch(w, () -> elbo_time) for w in workers() ]
elbo_time / maximum(elbo_times)
elbo_time / nw
num_sources = [ remotecall_fetch(w, () -> length(node_sources())) for w in workers() ]

elbo_times ./ num_sources

result_filename = joinpath(dat_dir, "parallel_results_$(int(time())).JLD")
result_dict = Dict()
result_dict["elbo_time"] = elbo_time;
result_dict["elbo_times"] = elbo_times;
# result_dict["accum"] = accum;
# result_dict["accum_par"] = accum_par;
result_dict["nw"] = nw;
result_dict["frame_jld_file"] = frame_jld_file;
result_dict["synthetic"] = synthetic;

JLD.save(result_filename, result_dict)
