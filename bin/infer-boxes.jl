#!/usr/bin/env julia

import Celeste.ParallelRun: BoundingBox, infer_boxes
import Celeste.Log
using Celeste: SDSSIO

include("binutil.jl")

function run_infer_boxes(args::Vector{String})
    if length(args) < 3
        println("""
Usage:
  infer-boxes.jl [--noprefetch] [--iostrategy=<strategy>] <rcf_nsrcs_file> <boxes_file> [<boxes_file>...] <out_dir>

Supported IO Strategies (default is FITS):
  - fits: Load data from stagedir in original SDSS FITS format
  - mdtfits: Load data from stagedir, split over 5 mdt directories
  - bigfiles: Load data in bigfile format

<rcf_nsrcs_file> format, one line per RCF:
  <run>	<camcol>	<field>	<num_primary_sources>

<boxes_file> format, one line per box:
  <difficulty>	<#RCFs>	<#sources>	<ramin> <ramax> <decmin> <decmax>	<rcf1idx>,<rcf2idx>...
            """)
        exit(-1)
    end

    prefetch = true
    if args[1] == "--noprefetch"
        shift!(args)
        prefetch = false
    end

    strategyarg = ""
    if startswith(args[1], "--iostrategy=")
        strategyarg = shift!(args)[length("--iostrategy=")+1:end]
    end

    strategy, all_rcfs, all_rcf_nsrcs = decide_strategy(strategyarg, args[1])

    # parse the specified box file(s)
    nboxfiles = length(args) - 2
    all_boxes = Vector{Vector{BoundingBox}}()
    all_boxes_rcf_idxs = Vector{Vector{Vector{Int32}}}()
    for i = 1:nboxfiles
        boxfile = args[i+1]
        boxes = Vector{BoundingBox}()
        boxes_rcf_idxs = Vector{Vector{Int32}}()
        f = open(boxfile)
        for ln in eachline(f)
            lp = split(ln, '\t')
            if length(lp) != 5
                Log.one_message("ERROR: malformed line in box file:\n> $ln ")
                continue
            end

            ss = split(lp[4], ' ')
            ramin = parse(Float64, ss[1])
            ramax = parse(Float64, ss[2])
            decmin = parse(Float64, ss[3])
            decmax = parse(Float64, ss[4])
            bb = BoundingBox(ramin, ramax, decmin, decmax)
            push!(boxes, bb)

            ris = split(lp[5], ',')
            rcf_idxs = [parse(Int32, ri) for ri in ris]
            push!(boxes_rcf_idxs, rcf_idxs)
        end
        close(f)
        if length(boxes) < 1
            Log.one_message("$boxfile is empty?")
            continue
        end
        push!(all_boxes, boxes)
        push!(all_boxes_rcf_idxs, boxes_rcf_idxs)
    end
    if length(all_boxes) < 1
        Log.one_message("box file(s) empty?")
        exit(-3)
    end

    infer_boxes(all_rcfs, all_rcf_nsrcs, all_boxes, all_boxes_rcf_idxs,
                strategy, prefetch, args[end])
end

run_infer_boxes(ARGS)
