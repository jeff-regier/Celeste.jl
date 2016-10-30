#!/usr/bin/env julia

import Celeste.ParallelRun: infer_rcf
import Celeste.SDSSIO: RunCamcolField


const usage_info =
"""
Usage:
  infer-rcf <run> <camcol> <field>
"""

const stagedir = ENV["CELESTE_STAGE_DIR"]

if length(ARGS) != 3
    println(usage_info)
else
    rcf = RunCamcolField(ARGS...)
    # Output gets written to the top level of the staging directory.
    # We may want to modify that in the future by changing the third argument.
    infer_rcf(rcf, stagedir, stagedir)
end

