#!/usr/bin/env julia

import Celeste.ParallelRun: BoundingBox, get_overlapping_fields,
                            one_node_joint_infer, infer_init, find_neighbors
import Celeste.SDSSIO: RunCamcolField, load_field_images, PlainFITSStrategy


const rcfs = [
    RunCamcolField(4264,6,160),
    RunCamcolField(4264,6,161),
    RunCamcolField(4264,5,158),
    RunCamcolField(4264,5,159),
    RunCamcolField(4264,5,160),
    RunCamcolField(4264,5,161),
    RunCamcolField(4264,5,162),
    RunCamcolField(4294,6,133),
    RunCamcolField(4294,6,134),
    RunCamcolField(4294,6,135),
    RunCamcolField(4294,6,136),
    RunCamcolField(4294,6,137),
    RunCamcolField(4294,6,138),
    RunCamcolField(4294,5,135)]


const datadir = joinpath(dirname(@__FILE__), "..", "..", "test", "data")
wd = pwd()
cd(datadir)
for rcf in rcfs
    run(`make RUN=$(rcf.run) CAMCOL=$(rcf.camcol) FIELD=$(rcf.field)`)
end
cd(wd)

function benchmark_quarter_degree()
    box = BoundingBox(124.0, 124.5, 58.5, 59.0)
    rcfs = get_overlapping_fields(box, datadir)

    # ctni = (catalogs, target, neighbor_map, images)
    ctni = infer_init(rcfs, PlainFITSStrategy(datadir); box=box)[1:4]

    # Warm up---this compiles the code
    ctni2 = (ctni[1], ctni[2][1:1], ctni[3][1:1], ctni[4][1:1])
    @time one_node_joint_infer(ctni2...)
    println("Done with warm up")

    # resets runtime profiler *and* count for --track-allocation
    Profile.clear_malloc_data()

    if isempty(ARGS)
        @time  one_node_joint_infer(ctni...)
    elseif ARGS[1] == "--profile"
        Profile.init(delay=1.0)
        @profile one_node_joint_infer(ctni...)
        Profile.print(format=:flat, sortedby=:count)
    end
end


benchmark_quarter_degree()
