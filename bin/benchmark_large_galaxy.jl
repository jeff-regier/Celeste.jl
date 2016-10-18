#!/usr/bin/env julia

import Celeste: ParallelRun
import Celeste.SDSSIO: RunCamcolField


const datadir = joinpath(Pkg.dir("Celeste"), "test", "data")
wd = pwd()
cd(datadir)
run(`make`)
run(`make RUN=3893 CAMCOL=2 FIELD=261`)
cd(wd)


function benchmark_infer()
    # this box contains a couple of small sources--just use it to compile the code
    box_compile = ParallelRun.BoundingBox(164.40, 164.41, 39.12, 39.125)
    field_triplets_compile = [RunCamcolField(3900, 6, 269),]
    ParallelRun.one_node_infer(field_triplets_compile, datadir; box=box_compile)

    # this box contains just objid 1237662193995284525--a really big galaxy!
    box = ParallelRun.BoundingBox(202.1093, 202.1094, 40.7294, 40.7296)
    field_triplets = [RunCamcolField(3893, 2, 261),]
    @time ParallelRun.one_node_infer(field_triplets, datadir; box=box)
    @time ParallelRun.one_node_infer(field_triplets, datadir; box=box)
end


benchmark_infer()
