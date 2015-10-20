#!/usr/bin/env julia

# Use nw workers.
nw = 3
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
  frame_jld_file = "initialzed_celeste_003900-6-0269.JLD"
  S = 10
  synthetic = true
end


load_cluster_data();
initialize_cluster();

@everywhere begin
  # Shrink the radii artificially.
  scale_patch_size = 0.2
  mp = ModelInit.initialize_model_params(
        tiled_blob, blob, body, fit_psf=true, scale_patch_size=scale_patch_size)
end

b = 3
has_source = Bool[ 1 in mp.tile_sources[b][i, j]
  for i in 1:size(mp.tile_sources[b])[1], j in 1:size(mp.tile_sources[b])[2] ];
sum(has_source)


# Sanity check that the tiles were communicated successfully
@everywhere tilesum = sum([sum([ sum(t.pixels) for t in tiled_blob[b]]) for b=1:5 ])
remote_tilesums = [remotecall_fetch(w, () -> tilesum) for w in workers()];
@assert tilesum == sum(remote_tilesums)

# Sanity check that the mp is the same.
for w in workers()
  @assert mp.vp == remotecall_fetch(w, () -> mp.vp)
end


# Set up a transform
@everywhere begin
  omitted_ids = Int64[];
  worker_sources = node_sources();
  transform = Transform.get_mp_transform(mp);
  x = transform.vp_to_array(mp.vp, omitted_ids);
  k = size(x)[1]
  mp_dual = CelesteTypes.convert(ModelParams{DualNumbers.Dual}, mp);
end

@everywhere begin
  hess_i, hess_j, hess_val, new_hess_time =
    OptimizeElbo.elbo_hessian(tiled_blob, x, mp_dual, transform,
                              omitted_ids, verbose=true);
  new_hess_sparse =
    OptimizeElbo.unpack_hessian_vals(hess_i, hess_j, hess_val, size(x));
end









#######################################
# Evaluate the elbo.  Do it twice to avoid compile time.

# locally:
accum = zero_sensitive_float(CanonicalParams, Float64, mp.S);
@time elbo_time = eval_worker_likelihood()
@time elbo_time = eval_worker_likelihood()

# Evaluate the ELBO in parallel.  Most of the time is taken up on the workers,
# not in communication.
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
