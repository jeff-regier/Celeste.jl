#!/usr/bin/env julia

import Celeste: ParallelRun
import Celeste.SDSSIO: RunCamcolField


const datadir = joinpath(Pkg.dir("Celeste"), "test", "data")
wd = pwd()
cd(datadir)
run(`make RUN=3900 CAMCOL=2 FIELD=453`)
run(`make RUN=4382 CAMCOL=1 FIELD=70`)
cd(wd)


function benchmark_infer()
    # Warm up---this compiles the code
    box_compile = ParallelRun.BoundingBox(164.39, 164.40, 39.11, 39.13)
    field_triplets_compile = [RunCamcolField(3900, 6, 269),]
    ParallelRun.one_node_infer(field_triplets_compile, datadir; box=box_compile)

    # This box should only contain objid = 1237662224072638610, a faint
    # galaxy that overlaps with a really bright galaxy.
    # See http://skyserver.sdss.org/dr10/en/tools/chart/navi.aspx?ra=200.00958464398&dec=38.1880678852474
    # Update: Looks like the run time is all PSF fitting.
    # That's not worth fixing since PSF fitting is going away.
    # Let's update this script with another objid that's slow once we come across one.
    box = ParallelRun.BoundingBox(200.00957, 200.00959, 38.18806, 38.18807)
    field_triplets = [RunCamcolField(3900, 2, 453), RunCamcolField(4382, 1, 70)]
    @time ParallelRun.one_node_infer(field_triplets, datadir; box=box)

    Profile.init(10^6, 0.01)
    Profile.clear_malloc_data()
    # about half the run time is psf fitting, the other half is elbo evaluation
    @profile ParallelRun.one_node_infer(field_triplets, datadir; box=box)
    Profile.print(format=:flat, sortedby=:count)
end


benchmark_infer()
