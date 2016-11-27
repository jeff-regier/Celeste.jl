#!/usr/bin/env julia

import Celeste.ParallelRun: one_node_single_infer, infer_init, BoundingBox
import Celeste.SDSSIO: RunCamcolField, load_field_images
import Celeste.Infer: find_neighbors


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
    box = BoundingBox(164.39, 164.41, 39.11, 39.13)
    rcfs = [RunCamcolField(3900, 6, 269),]

    catalog, target_sources = infer_init(rcfs, datadir; box=box)
    images = load_field_images(rcfs, datadir)
    neighbor_map = find_neighbors(target_sources, catalog, images)

    # Warm up---this compiles the code
    one_node_single_infer(catalog, target_sources, neighbor_map, images)

    # clear allocations in case julia is running with --track-allocations=user
    Profile.clear_malloc_data()

    if isempty(ARGS)
        @time one_node_single_infer(catalog, target_sources, neighbor_map, images)
    elseif ARGS[1] == "--profile"
        Profile.clear_malloc_data()
        # about half the run time is psf fitting, the other half is elbo evaluation
        @profile one_node_single_infer(catalog, target_sources, neighbor_map, images)
        Profile.print(format=:flat, sortedby=:count)
    end
end


benchmark_infer()
