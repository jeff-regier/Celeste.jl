#!/usr/bin/env julia

import Celeste.ParallelRun: BoundingBox, get_overlapping_fields,
                            one_node_infer, one_node_joint_infer
import Celeste.SDSSIO: RunCamcolField
import Celeste.DeterministicVIImagePSF: infer_source_fft
import Celeste.DeterministicVI: infer_source


const rcfs = [
    RunCamcolField(4294,6,136),
    RunCamcolField(4264,5,158),
    RunCamcolField(4264,5,159),
    RunCamcolField(4264,5,161),
    RunCamcolField(4264,6,160),
    RunCamcolField(4264,5,160),
    RunCamcolField(4294,6,135),
    RunCamcolField(4294,5,135)]

const datadir = joinpath(Pkg.dir("Celeste"), "test", "data")
wd = pwd()
cd(datadir)
for rcf in rcfs
    run(`make RUN=$(rcf.run) CAMCOL=$(rcf.camcol) FIELD=$(rcf.field)`)
end
cd(wd)

"""
This benchmark optimizes all the light sources in a
one-sixteenth-square-degree region of sky.
During the optimization, a pixel is visited
35,937,971 times (``pixel visits'').
"""
function benchmark_sixteenth_degree()
    box = BoundingBox(124.25, 124.50, 58.5, 58.75)

    wrap_joint(cnti...) = one_node_joint_infer(cnti...; use_fft=false)

    warmup_box = BoundingBox(124.25, 124.26, 58.7, 58.71)
    warmup_rcfs = get_overlapping_fields(warmup_box, datadir)
    one_node_infer(warmup_rcfs,
                   datadir;
                   infer_callback=wrap_joint,
                   box=warmup_box)

    rcfs = get_overlapping_fields(box, datadir)

    # resets runtime profiler *and* count for --track-allocation
    Profile.clear_malloc_data()

    if isempty(ARGS)
        @time one_node_infer(rcfs, datadir; infer_callback=wrap_joint, box=box)
    elseif ARGS[1] == "--profile"
        Profile.init(delay=1.0)
        @profile one_node_infer(rcfs, datadir; infer_callback=wrap_joint, box=box)
        Profile.print(format=:flat, sortedby=:count)
    end
end


benchmark_sixteenth_degree()
