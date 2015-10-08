# Calculate values and partial derivatives of the variational ELBO.

# For various reasons, e.g. the need to access the global scope when
# communicating to subprocesses, this is best not done as a module.
#module CelesteCluster

VERSION < v"0.4.0-dev" && using Docile
using CelesteTypes
using JLD

using ElboDeriv.tile_likelihood!
using ElboDeriv.SourceBrightness
using ElboDeriv.BvnComponent
using ElboDeriv.GalaxyCacheComponent

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
A type containing all the information that needs to be communicated
to worker nodes at each iteration.

Attributes:
  vp: The VariationalParams for the ModelParams object
  star_mcs_vec: A vector of star BVN components, one for each band
  gal_mcs_vec: A vector of galaxy BVN components, one for each band
  sbs_vec: A vector of brightness vectors, one for each band
""" ->
type ParamState{NumType <: Number}
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


@doc """
Return a vector of the sources that affect the node.
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
Update param_state in place using mp.
""" ->
function update_param_state!{NumType <: Number}(
    mp::ModelParams{NumType}, param_state::ParamState{NumType})
    for b=1:5
      star_mcs, gal_mcs = ElboDeriv.load_bvn_mixtures(mp, b);
      sbs = ElboDeriv.SourceBrightness{Float64}[
        ElboDeriv.SourceBrightness(mp.vp[s]) for s in 1:mp.S];

      param_state.star_mcs_vec[b] = star_mcs
      param_state.gal_mcs_vec[b] = gal_mcs
      param_state.sbs_vec[b] = sbs
    end
end


function create_workers(nw::Int64)
  # Close all but the current worker and then add nw of them.
  println("Adding workers.")
  for worker in workers()
    if worker != myid()
      rmprocs(worker)
    end
  end
  addprocs(nw)
  @assert(length(workers()) == nw)
end


# This requires synthetic and frame_jld_file to be defined
# everywhere in the global scope.
function load_cluster_data()
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
end

@doc """
Make sure there are <nw> workers.  Intended to be called on startup only.

Requires mp to be defined in the global scope and for every
worker to have a tiled_blob defined.
""" ->
function initialize_cluster()
  # Divide up the tiled_blobs.
  # Unfortunately, we run out of memory
  # when we try to communciate actual subsets of a tiled_blob over sockets,
  # so currently each node must load the whole file and then subset it.

  # For now it's more convenient to define global variables than
  # to make RemoteRefs for everything.
  global col_ranges
  global worker_ids
  global param_state
  global mp
  global tiled_blob
  global param_state_rr

  println("Dividing the blobs.")
  for b=1:5
    #global col_ranges
    col_cuts = iround(linspace(1, size(tiled_blob[b])[2] + 1, nw + 1))
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
      tiled_blob[b] = tiled_blob[b][:, col_ranges[proc_id]]
    end
  end

  # Initialize the ParamState sockets and copy the ModelParams.
  println("Initializing the parameters.")
  param_state = ParamState(mp);
  update_param_state!(mp, param_state);

  @everywhereelse begin
    global param_state_rr
    global mp
    mp = remotecall_fetch(1, () -> mp);
    param_state_rr = RemoteRef(1)
  end
  param_state_rr = [remotecall_fetch(w, () -> param_state_rr) for w in workers()]
  for rr in param_state_rr
    put!(rr, param_state)
  end
  @everywhereelse param_state = fetch(param_state_rr)

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

  param_state_rr, accum_rr
end


@doc """
Helper for remote evaluation of the elbo.
""" ->
function eval_likelihood!{NumType <: Number}(
    accum::SensitiveFloat{CanonicalParams, NumType},
    param_state::ParamState{NumType}, mp::ModelParams{NumType},
    tiled_blob::TiledBlob)

  println(myid(), " is starting.")
  elbo_time = time()
  global accum
  clear!(accum)
  mp.vp = param_state.vp
  for b in 1:5
    #println("Proc $(myid()) starting band $b")
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
  elbo_time
end

function eval_likelihood()
  global accum
  global param_state
  global mp
  global tiled_blob

  eval_likelihood!(accum, param_state, mp, tiled_blob)
end

#end # Module end
