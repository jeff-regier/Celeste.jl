#!/usr/bin/env julia

import Celeste.ParallelRun: BoundingBox, estimate_box_runtime
import Celeste.Log


const usage_info =
"""
Usage:
  estimate-box-runtime.jl <ramin> <ramax> <decmin> <decmax>
"""

const stagedir = ENV["CELESTE_STAGE_DIR"]

if length(ARGS) != 4
    println(usage_info)
else
    box = BoundingBox(ARGS...)
    num_rcfs, num_targets = estimate_box_runtime(box, stagedir)
    box_str = "$(box.ramin) $(box.ramax) $(box.decmin) $(box.decmax)"
    Log.info("$(num_rcfs * num_targets) hardness ($num_rcfs rcfs x $num_targets targets) for region $box_str")
end

