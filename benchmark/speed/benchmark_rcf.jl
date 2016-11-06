#!/usr/bin/env julia

import Celeste: ParallelRun
import Celeste.SDSSIO: RunCamcolField


const rcfs = [
    RunCamcolField(3900,1,177),
    RunCamcolField(3900,1,178),
    RunCamcolField(3900,1,179),
    RunCamcolField(3900,2,178),
    RunCamcolField(4469,1,251),
    RunCamcolField(4469,1,252),
    RunCamcolField(4469,1,253),
    RunCamcolField(4469,1,254),
    RunCamcolField(4392,6,36),
    RunCamcolField(4392,6,37),
    RunCamcolField(4392,6,38),
    RunCamcolField(4516,6,148),
    RunCamcolField(4516,6,149),
    RunCamcolField(4516,6,150),
    RunCamcolField(4518,6,145),
    RunCamcolField(4518,6,146),
    RunCamcolField(4518,6,147),
    RunCamcolField(3900,1,177),
    RunCamcolField(3900,1,178),
    RunCamcolField(3900,1,179),
    RunCamcolField(3900,2,178),
    RunCamcolField(4469,1,251),
    RunCamcolField(4469,1,252),
    RunCamcolField(4469,1,253),
    RunCamcolField(4469,1,254),
    RunCamcolField(4392,6,36),
    RunCamcolField(4392,6,37),
    RunCamcolField(4392,6,38),
    RunCamcolField(4516,6,148),
    RunCamcolField(4516,6,149),
    RunCamcolField(4516,6,150),
    RunCamcolField(4518,6,145),
    RunCamcolField(4518,6,146),
    RunCamcolField(4518,6,147)]

const datadir = joinpath(Pkg.dir("Celeste"), "test", "data")
wd = pwd()
cd(datadir)
for rcf in rcfs
    run(`make RUN=$(rcf.run) CAMCOL=$(rcf.camcol) FIELD=$(rcf.field)`)
end
cd(wd)

"""
This benchmark optimizes all the light sources listed as primary detections
for a particular "target" RCF. (Primary detections are unique--no light
source is a primary detection in more than one RCF.)
This RCF has only 30 light sources in it, whereas 400 light sources
is more standard. So this benchmark overstates the cost of loading images
as a proportion of total runtime. On the other hand, in production, 
loading sources is single threaded, whereas optimizing sources is
multithreaded.
"""
function benchmark_infer_rcf()
    # This is the "target" run/camcol/field (RCF).
    rcf = RunCamcolField(4518,6,146)

    # Warm up---this compiles the code
    ParallelRun.infer_rcf(rcf, datadir, datadir; objid="1237664880489791577")

    # resets runtime profiler *and* count for --track-allocation
    Profile.clear_malloc_data()

    if isempty(ARGS)
        @time ParallelRun.infer_rcf(rcf, datadir, datadir)
    elseif ARGS[1] == "--profile"
        @profile ParallelRun.infer_rcf(rcf, datadir, datadir)
        Profile.print(format=:flat, sortedby=:count)
    end
end


benchmark_infer_rcf()
