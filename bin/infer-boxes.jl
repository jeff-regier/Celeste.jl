#!/usr/bin/env julia

import Celeste.ParallelRun: BoundingBox, infer_boxes
import Celeste.Log


const usage_info =
"""
Usage:
  infer-boxes.jl <boxes_file> <out_dir>
"""

const stagedir = ENV["CELESTE_STAGE_DIR"]

if length(ARGS) != 2
    println(usage_info)
else
    all_boxes = BoundingBox[]
    sky_boxes = ARGS[1]
    f = open(sky_boxes)
    for ln in eachline(f)
        ss = split(ln, ' ')
        ns = [parse(Float64, x) for x in ss]
        bb = BoundingBox(ns...)
        push!(all_boxes, bb)
    end
    close(f)
    outdir = ARGS[2]
    infer_boxes(all_boxes, stagedir, outdir)
end

