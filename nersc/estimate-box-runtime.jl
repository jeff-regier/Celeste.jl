#!/usr/bin/env julia

import Celeste.ParallelRun: BoundingBox, get_overlapping_fields, infer_init
import Celeste: Log, SDSSIO


const usage_info =
"""
Usage:
  estimate-box-runtime.jl <ramin> <ramax> <decmin> <decmax>
"""

const stagedir = ENV["CELESTE_STAGE_DIR"]

const max_hardness = 10_000

function to_tasks(box)
    rcfs = get_overlapping_fields(box, stagedir)
    catalog = SDSSIO.read_photoobj_files(rcfs, stagedir,
                        duplicate_policy=:primary)
    entry_in_range = entry->((box.ramin < entry.pos[1] < box.ramax) &&
                             (box.decmin < entry.pos[2] < box.decmax))
    target_sources = find(entry_in_range, catalog)

    num_rcfs, num_targets = length(rcfs), length(target_sources)

    hardness = num_rcfs * num_targets
    if hardness <= max_hardness
        box_str = "$(box.ramin) $(box.ramax) $(box.decmin) $(box.decmax)"
        println("$hardness\t$num_rcfs\t$num_targets\t$box_str")
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

