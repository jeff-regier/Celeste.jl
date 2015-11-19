# Distribute Celeste calculations across a cluster of workers.

# For various reasons, e.g. the need to access the global scope when
# communicating to subprocesses, I have not made this into a module.

VERSION < v"0.4.0-dev" && using Docile
using Celeste
using CelesteTypes
using JLD
import SampleData.dat_dir

# Like @everywhere but for remote workers only.
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


@doc """
Return a vector of the sources that affect the node.  Assumes tiled_blob
and mp are globally defined.
""" ->
function node_sources()
  sources = Int64[]
  for b in 1:5
    for tile in tiled_blob[b][:]
      append!(sources, mp.tile_sources[b][tile.hh, tile.ww])
    end
  end
  unique(sources)
end

@doc """
Close all but the current worker and then add nw of them.
""" ->
function create_workers(nw::Int64)
  println("Adding workers.")
  for worker in workers()
    if worker != myid()
      rmprocs(worker)
    end
  end
  addprocs(nw)
  @assert(length(workers()) == nw)
end


@doc """
Load synthetic or actual data across the cluster of workers.

This requires certain variables to be defined in the global scope.
- synthetic
- If synthetic, then S
- If !synthetic, then dat_dir and frame_jld_file.

If synthetic is false, the file frame_jld_file must contain a dictionary
with an initialized TiledBlob and ModelParams object.
""" ->
function load_cluster_data()
  println("Loading data.")
  @everywhere begin
    if synthetic
      srand(1)
      blob, original_mp, body, tiled_blob =
        SampleData.gen_n_body_dataset(S, tile_width=10);
    else
      img_dict = JLD.load(joinpath(dat_dir, frame_jld_file));
      tiled_blob = img_dict["tiled_blob"];
      original_mp = img_dict["mp_all"];
    end
    mp = deepcopy(original_mp);
  end;
end


@doc """
Perform a number of initialization tasks across the cluster.  The result
is a set of globally defined variables on each worker.  It assumes that
each worker has its own identical copy of mp and tiled_blob as global
variables.

The tasks are:
- Divide up the blobs by assigning to each worker a unique subset of the
  tile columns.
- Initialize the ParameterMessage objects and define RemoteRefs for
  communicating updates.
- Initialize the sensitive floats and define RemoteRefs for communicating them.
""" ->
function initialize_cluster()
  # It's more convenient to define global variables than
  # to make RemoteRefs for everything.  Eventually maybe we should wrap
  # everything into a single type with its own RemoteRef?
  global col_ranges
  global worker_ids
  global param_msg
  global mp
  global tiled_blob
  global param_msg_rr

  # Divide up the tiled_blobs.
  # Unfortunately, we run out of memory
  # when we try to communciate actual subsets of a tiled_blob over sockets,
  # so currently each node must load the whole file and then subset it.
  println("Dividing the blobs.")
  for b=1:5
    col_cuts = round(Integer, linspace(1, size(tiled_blob[b])[2] + 1, nw + 1))
    col_ranges = map(i -> col_cuts[i]:col_cuts[i + 1] - 1, 1:nw)
  end

  worker_ids = workers()
  @everywhereelse begin
    worker_ids = remotecall_fetch(1, () -> worker_ids)
    proc_id_vec = find(worker_ids .== myid())
    @assert length(proc_id_vec) == 1
    proc_id = proc_id_vec[1]
    col_ranges = remotecall_fetch(1, () -> col_ranges)
    for b=1:5
      # TODO: You should make a tiled_blob slicing function.
      tiled_blob[b] = tiled_blob[b][:, col_ranges[proc_id]]
    end
  end

  # Initialize the ParameterMessage sockets and copy the ModelParams.
  println("Initializing the parameters.")
  param_msg = ElboDeriv.ParameterMessage(mp);
  ElboDeriv.update_parameter_message!(mp, param_msg);

  @everywhereelse begin
    global param_msg_rr
    global mp
    mp = remotecall_fetch(1, () -> mp);
    param_msg_rr = RemoteRef(1)
  end
  param_msg_rr = [remotecall_fetch(w, () -> param_msg_rr) for w in workers()]
  for rr in param_msg_rr
    put!(rr, param_msg)
  end
  @everywhereelse param_msg = fetch(param_msg_rr)

  # Set up for accum to be communicated back to process 1
  println("Initializing the accum sockets.")
  global accum_rr
  accum_rr = [ RemoteRef(w) for w in workers() ]
  @everywhereelse begin
    accum = zero_sensitive_float(CanonicalParams, Float64, mp.S)
    accum_rr = RemoteRef(myid())
    accum_rr = remotecall_fetch(1, i -> accum_rr[i], proc_id[1])
    put!(accum_rr, accum)
  end

  param_msg_rr, accum_rr
end


function eval_worker_likelihood()
  global tiled_blob
  global param_msg
  global mp
  global accum

  println("Worker ", myid(), " is starting the likelihood evaluation.")
  elbo_time = time()
  ElboDeriv.elbo_likelihood!(tiled_blob, param_msg, mp, accum)
  elbo_time = time() - elbo_time
  println("Worker ", myid(), " is done in $(elbo_time)s.")
  elbo_time
end



function eval_worker_hessian()
  omitted_ids = Int64[]

  global hess_i, hess_j, hess_val, new_hess_time

  # TODO: Only use an x containing the local sources?
  x = transform.vp_to_array(mp.vp, omitted_ids);

  mp_dual = CelesteTypes.convert(ModelParams{DualNumbers.Dual{Float64}}, mp);
  @time hess_i, hess_j, hess_val, new_hess_time =
    OptimizeElbo.elbo_hessian(tiled_blob, x, mp_dual, transform,
                              omitted_ids, verbose=true,
                              deriv_sources=worker_sources);
end
