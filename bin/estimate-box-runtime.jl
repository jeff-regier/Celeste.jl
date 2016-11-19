#!/usr/bin/env julia

import Celeste.ParallelRun: BoundingBox, estimate_box_runtime


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
    num_active = estimate_box_runtime(box, stagedir)
    println("The specified region contains $num_active active pixels.")
end

