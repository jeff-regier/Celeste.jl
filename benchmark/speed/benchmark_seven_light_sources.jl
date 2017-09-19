#!/usr/bin/env julia

import Celeste.ParallelRun: one_node_joint_infer, infer_init, BoundingBox, find_neighbors
import Celeste.SDSSIO: RunCamcolField, load_field_images, PlainFITSStrategy

const datadir = joinpath(dirname(@__FILE__), "..", "..", "test", "data")
wd = pwd()
cd(datadir)
run(`make`)
cd(wd)

"""
This benchmark operates on a box of the sky that contains
seven light sources. During the optimization, some pixel is visited
254,771 times (``pixel visits'').
"""
function benchmark_seven_light_sources()
    # very small patch of sky that turns out to have 4 sources.
    # We checked that this patch is in the given field.
    box = BoundingBox(164.39, 164.41, 39.11, 39.13)
    rcfs = [RunCamcolField(3900, 6, 269),]

    catalog, target_sources, neighbor_map, images =
                        infer_init(rcfs, PlainFITSStrategy(datadir); box=box)
    ctni = (catalog, target_sources, neighbor_map, images)

    # Warm up---this compiles the code
    ctni2 = (catalog, target_sources[1:1], neighbor_map[1:1], images[1:1])
    one_node_joint_infer(ctni2...)

    # clear allocations in case julia is running with --track-allocations=user
    Profile.clear_malloc_data()

    if isempty(ARGS)
        @time one_node_joint_infer(ctni...)
    elseif ARGS[1] == "--profile"
        Profile.init(delay=0.01)
        @profile one_node_joint_infer(ctni...)
        Profile.print(format=:flat, sortedby=:count)
    end
end


benchmark_seven_light_sources()
