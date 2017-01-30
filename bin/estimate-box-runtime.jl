#!/usr/bin/env julia

import Celeste.ParallelRun: BoundingBox, get_overlapping_fields, infer_init
import Celeste.Log


const usage_info =
"""
Usage:
  estimate-box-runtime.jl <ramin> <ramax> <decmin> <decmax>
"""

const stagedir = ENV["CELESTE_STAGE_DIR"]

const max_hardness = 100_000

function to_tasks(box)
    rcfs = get_overlapping_fields(box, stagedir)
    catalog, target_sources = infer_init(rcfs, stagedir; box=box)
    num_rcfs, num_targets = length(rcfs), length(target_sources)

    hardness = num_rcfs * num_targets
    if hardness <= max_hardness
        box_str = "$(box.ramin) $(box.ramax) $(box.decmin) $(box.decmax)"
        Log.info(string("$(hardness) hardness ($num_rcfs rcfs x $num_targets ",
                        "targets) for region $box_str"))
    else
        sl = (box.ramax - box.ramin) / 2
        to_tasks(BoundingBox(box.ramin, box.ramin + sl,
                             box.decmin, box.decmin + sl))
        to_tasks(BoundingBox(box.ramin + sl, box.ramax,
                             box.decmin, box.decmin + sl))
        to_tasks(BoundingBox(box.ramin, box.ramin + sl,
                             box.decmin + sl , box.decmax))
        to_tasks(BoundingBox(box.ramin + sl, box.ramax,
                             box.decmin + sl, box.decmax))
    end
end


if length(ARGS) != 4
    println(usage_info)
else
    to_tasks(BoundingBox(ARGS...))
end

