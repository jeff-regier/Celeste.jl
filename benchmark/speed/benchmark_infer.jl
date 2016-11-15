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
    # Warm up---this compiles the code
    box_compile = ParallelRun.BoundingBox(164.39, 164.40, 39.11, 39.13)
    field_triplets_compile = [RunCamcolField(3900, 6, 269),]
    ParallelRun.one_node_infer(field_triplets_compile, datadir; box=box_compile)

    # clear allocations in case julia is running with --track-allocations=user
    Profile.clear_malloc_data()

    # very small patch of sky that turns out to have 4 sources.
    # We checked that this patch is in the given field.
    box = ParallelRun.BoundingBox(164.39, 164.41, 39.11, 39.13)
    field_triplets = [RunCamcolField(3900, 6, 269),]

    if isempty(ARGS)
        # takes 6.4 seconds as of 11/5/2016 on an Intel Core i5-6600 processor
        @time ParallelRun.one_node_infer(field_triplets, datadir; box=box)
    elseif ARGS[1] == "--profile"
        Profile.clear_malloc_data()
        # about half the run time is psf fitting, the other half is elbo evaluation
        @profile ParallelRun.one_node_infer(field_triplets, datadir; box=box)
        Profile.print(format=:flat, sortedby=:count)
    end
end


benchmark_infer()
