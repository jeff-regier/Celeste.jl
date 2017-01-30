#!/usr/bin/env julia

import Celeste.ParallelRun: BoundingBox, get_overlapping_fields
import Celeste.SDSSIO: RunCamcolField


# Lists all Run-Camcol-Field triplets that overlap with a specified
# bounding box. Output is in a format that can be piped to make, with the
# makefile in this directory, e.g.
#
#    ./list_rcfs.jl -999 999 -999 999 | sort -R | xargs -P 32 -n 1 make

if length(ARGS) == 4
    box = BoundingBox(ARGS...)
    rcfs = get_overlapping_fields(box, dirname(ENV["FIELD_EXTENTS"]))

    for rcf in rcfs
        println("RUN=$(rcf.run) CAMCOL=$(rcf.camcol) FIELD=$(rcf.field)")
    end
else
    println("Usage:\n    list_rcfs.jl <ramin> <ramax> <decmin> <decmax>")
end
