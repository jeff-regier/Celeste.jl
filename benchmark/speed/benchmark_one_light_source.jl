#!/usr/bin/env julia

import Celeste.ParallelRun: one_node_joint_infer, infer_init, BoundingBox
import Celeste.SDSSIO: RunCamcolField, load_field_images
import Celeste.Infer: find_neighbors
import Celeste.DeterministicVIImagePSF: infer_source_fft
import Celeste.DeterministicVIImagePSF

const datadir = joinpath(Pkg.dir("Celeste"), "test", "data")
wd = pwd()
cd(datadir)
run(`make RUN=7713 CAMCOL=3 FIELD=152`)
cd(wd)

"""
This benchmark operates on a box of the sky that contains just
one light source. It visits 1048 pixels per evaluation of the elbo,
and the optimizer runs for 37 iterations, for 38,776 pixel-visits
in total.
"""
function benchmark_one_light_source()
    box = BoundingBox(347.7444, 347.7446, 16.6202, 16.6204)
    rcfs = [RunCamcolField(7713, 3, 152)]

    catalog, target_sources = infer_init(rcfs, datadir; box=box)
    images = load_field_images(rcfs, datadir)
    neighbor_map = find_neighbors(target_sources, catalog, images)
    ctni = (catalog, target_sources, neighbor_map, images)

    # Warm up---this compiles the code
    one_node_joint_infer(ctni...; use_fft=true)

    # clear allocations in case julia is running with --track-allocations=user
    Profile.clear_malloc_data()

    if isempty(ARGS)
        @time one_node_joint_infer(ctni...; use_fft=true)
    elseif ARGS[1] == "--profile"
        Profile.init(delay=0.01)
        @profile one_node_joint_infer(ctni...; use_fft=true)
        Profile.print(format=:flat, sortedby=:count)
    end
end


benchmark_one_light_source()
