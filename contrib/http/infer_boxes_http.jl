# This is a demo script that shows how to use Celeste on a small
# cluster without a shared file system. It serves files over HTTP
# using nginx. It is appropriate for applications where network
# bandwidth significantly exceeds disk bandwith and for scales
# of < 1000 ranks.

# Note: This example assumes that SDSS data is available (in SDSS format) in
# /sdss on the master (the node from which this script is invoked). Different
# settings can be made in the simple_nginx.conf/iosettings.yml file.

usage = """
infer_boxes_http.jl <hostfile> <rcf_nsrcs> <boxfile>
"""

if length(ARGS) != 3
    println(usage)
    exit(-1)
end

hostfile = ARGS[1]
np = length(collect(eachline(hostfile)))

# Add 100 workers and initialize MPI between them
using MPI
manager=MPIManager(np=np, mpirun_cmd=`mpiexec -x JULIA_NUM_THREADS=8 -v -mca plm_rsh_no_tree_spawn 1 -np $np --map-by node --hostfile $hostfile`)
addprocs(manager)

# Load Celeste
using Celeste
@everywhere ENV["CELESTE_RANKS_PER_NODE"]=1
@everywhere ENV["CELESTE_THREADS_PER_CORE"]=1
here = @__DIR__
!isdefined(:CelesteMultiNode) && @everywhere include(joinpath($here,"..","..","src","multinode_run.jl"))
!isdefined(:CelesteHTTPIO) && @everywhere include(joinpath($here,"httpio.jl"))
include(joinpath(@__DIR__,"..","..","bin","binutil.jl"))

# Load Input settings
strategy = Celeste.read_settings_file(joinpath(@__DIR__,"iosettings.yml"))
all_rcfs, all_rcf_nsrcs = parse_rcfs_nsrcs(joinpath(ARGS[2]))
boxes, boxes_rcf_idxs = parse_boxfile(joinpath(ARGS[3]))

# Launch an nginx process for serving large static files
if isfile("/tmp/celeste-nginx.pid")
    run(pipeline(`kill -QUIT $(readstring("/tmp/celeste-nginx.pid")))`, stderr=DevNull))
end
spawn(`nginx -p $(@__DIR__) -c simple_nginx.conf`)

hostname = strip(readstring(`hostname`))

# Tell each worker to run Celeste
futures = map(workers()) do worker
    remotecall(Celeste.ParallelRun.infer_boxes, worker, CelesteMultiNode.DtreeStrategy(),
        all_rcfs, all_rcf_nsrcs, [boxes], [boxes_rcf_idxs],
        CelesteHTTPIO.HTTPStrategy("$hostname:10400", strategy), true, joinpath(ENV["HOME"],"output"))
end

# If we're not at the REPL, wait for completion before exiting
Base.isinteractive() || foreach(fetch, futures)
