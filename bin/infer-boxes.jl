#!/usr/bin/env julia

import Celeste.ParallelRun: BoundingBox, infer_boxes
import Celeste.Log
using Celeste: SDSSIO

distributed = haskey(ENV, "USE_DTREE")
distributed && !isdefined(:CelesteMultiNode) && include(joinpath(@__DIR__,"src","multinode_run.jl"))

include("binutil.jl")

function run_infer_boxes(args::Vector{String})
    if length(args) < 3
        println("""
Usage:
  infer-boxes.jl <settings> <rcf_nsrcs_file> <boxes_file> [<boxes_file>...] <out_dir>

<rcf_nsrcs_file> format, one line per RCF:
  <run>	<camcol>	<field>	<num_primary_sources>

<boxes_file> format, one line per box:
  <difficulty>	<#RCFs>	<#sources>	<ramin> <ramax> <decmin> <decmax>	<rcf1idx>,<rcf2idx>...
            """)
        exit(-1)
    end

    strategy = Celeste.read_settings_file(args[1])

    # load the RCFs #sources file
    all_rcfs, all_rcf_nsrcs = parse_rcfs_nsrcs(args[2])


    # parse the specified box file(s)
    nboxfiles = length(args) - 3
    all_boxes = Vector{Vector{BoundingBox}}()
    all_boxes_rcf_idxs = Vector{Vector{Vector{Int32}}}()
    for i = 1:nboxfiles
        all_boxes, all_boxes_rcf_idxs = parse_boxfile(args[i+2])
        push!(all_boxes, boxes)
        push!(all_boxes_rcf_idxs, boxes_rcf_idxs)
    end
    if length(all_boxes) < 1
        Log.one_message("box file(s) empty?")
        exit(-3)
    end

    infer_boxes(distributed ? Celeste.ParallelRun.ThreadsStrategy() :
                              CelesteMultiNode.DtreeStrategy(),
                all_rcfs, all_rcf_nsrcs,
                all_boxes, all_boxes_rcf_idxs,
                strategy, true, args[end])
end

run_infer_boxes(ARGS)
