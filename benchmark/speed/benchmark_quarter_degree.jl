#!/usr/bin/env julia

import Celeste.ParallelRun: BoundingBox, get_overlapping_fields,
                            one_node_infer, one_node_single_infer
import Celeste.SDSSIO: RunCamcolField
import Celeste.DeterministicVIImagePSF: infer_source_fft
import Celeste.DeterministicVI: infer_source

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

const datadir = joinpath(Pkg.dir("Celeste"), "test", "data")
wd = pwd()
cd(datadir)
for rcf in rcfs
    run(`make RUN=$(rcf.run) CAMCOL=$(rcf.camcol) FIELD=$(rcf.field)`)
end
cd(wd)

"""
This benchmark optimizes all the light sources in a
one-quarter-square-degree region of sky.
"""
function benchmark_quarter_degree()
    box = BoundingBox(124.0, 124.5, 58.5, 59.0)

    wrap_single(cnti...) = one_node_single_infer(cnti...;
                                  infer_source_callback=infer_source_fft)

    warmup_box = BoundingBox(124.2, 124.21, 58.7, 58.71)
    warmup_rcfs = get_overlapping_fields(warmup_box, datadir)
    one_node_infer(warmup_rcfs,
                   datadir;
                   infer_callback=wrap_single,
                   box=warmup_box)

    rcfs = get_overlapping_fields(box, datadir)

    # resets runtime profiler *and* count for --track-allocation
    Profile.clear_malloc_data()

    if isempty(ARGS)
        @time one_node_infer(rcfs, datadir; infer_callback=wrap_single, box=box)
    elseif ARGS[1] == "--profile"
        @profile one_node_infer(rcfs, datadir; infer_callback=wrap_single, box=box)
        Profile.print(format=:flat, sortedby=:count)
    end
end


benchmark_quarter_degree()
