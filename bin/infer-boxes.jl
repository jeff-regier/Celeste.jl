#!/usr/bin/env julia

import Celeste.ParallelRun: BoundingBox, infer_boxes
import Celeste.SDSSIO: RunCamcolField
import Celeste.Log


function run_infer_boxes(args::Vector{String})
    if length(args) < 3
        println("""
Usage:
  infer-boxes.jl <rcf_nsrcs_file> <boxes_file> [<boxes_file>...] <out_dir>

<rcf_nsrcs_file> format, one line per RCF:
  <run>	<camcol>	<field>	<num_primary_sources>

<boxes_file> format, one line per box:
  <difficulty>	<#RCFs>	<#sources>	<ramin> <ramax> <decmin> <decmax>	<rcf1idx>,<rcf2idx>...
            """)
        exit(-1)
    end
    if !haskey(ENV, "CELESTE_STAGE_DIR")
        Log.one_message("ERROR: set CELESTE_STAGE_DIR!")
        exit(-2)
    end

    # load the RCFs #sources file
    rcf_nsrcs_file = args[1]
    all_rcfs = Vector{RunCamcolField}()
    all_rcf_nsrcs = Vector{Int16}()
    f = open(rcf_nsrcs_file)
    for ln in eachline(f)
        lp = split(ln, '\t')
        run = parse(Int16, lp[1])
        camcol = parse(UInt8, lp[2])
        field = parse(Int16, lp[3])
        nsrc = parse(Int16, lp[4])
        push!(all_rcfs, RunCamcolField(run, camcol, field))
        push!(all_rcf_nsrcs, nsrc)
    end
    close(f)

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
        end
        push!(all_boxes, boxes)
        push!(all_boxes_rcf_idxs, boxes_rcf_idxs)
    end
    if length(all_boxes) < 1
        Log.one_message("box file(s) empty?")
        exit(-3)
    end

    infer_boxes(all_rcfs, all_rcf_nsrcs, all_boxes, all_boxes_rcf_idxs,
                ENV["CELESTE_STAGE_DIR"], args[end])
end

run_infer_boxes(ARGS)
