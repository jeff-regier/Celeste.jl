#!/usr/bin/env julia

import Celeste.ParallelRun: BoundingBox, infer_box


const usage_info =
"""
Usage:
  infer-box.jl <ramin> <ramax> <decmin> <decmax>
"""

const stagedir = ENV["CELESTE_STAGE_DIR"]

if length(ARGS) != 4
    println(usage_info)
else
    box = BoundingBox(ARGS...)
    # Output gets written to the top level of the staging directory.
    # We may want to modify that in the future by changing the third argument.
    @time infer_box(box, stagedir, stagedir)
end

