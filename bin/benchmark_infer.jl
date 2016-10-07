#!/usr/bin/env julia

import Celeste: ParallelRun
import Celeste.SDSSIO: RunCamcolField


const datadir = joinpath(Pkg.dir("Celeste"), "test", "data")
wd = pwd()
cd(datadir)
run(`make`)
cd(wd)


"""
test infer with a single (run, camcol, field).
This is basically just to make sure it runs at all.
"""
function benchmark_infer()
    # very small patch of sky that turns out to have 4 sources.
    # We checked that this patch is in the given field.
    box = ParallelRun.BoundingBox(164.39, 164.41, 39.11, 39.13)
    field_triplets = [RunCamcolField(3900, 6, 269),]
    @time ParallelRun.one_node_infer(field_triplets, datadir; box=box)

    # take 22 seconds on jeff's old desktop (intel core2 q6600 processor),
    # as of 10/7/2016
    @time ParallelRun.one_node_infer(field_triplets, datadir; box=box)

    Profile.init(10^8, 0.001)
    Profile.clear_malloc_data()
    # about half the run time is psf fitting, the other half is elbo evaluation
    @profile ParallelRun.one_node_infer(field_triplets, datadir; box=box)
    Profile.print(format=:flat, sortedby=:count)
end


benchmark_infer()
