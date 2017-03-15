#!/usr/bin/env julia

import Celeste.ParallelRun: BoundingBox, infer_boxes
import Celeste.Log

if length(ARGS) != 2
    println("""
        Usage:
          infer-boxes.jl <boxes_file> <out_dir>

        <boxes_file> format, one line per box:
        <difficulty>	<#RCFs>	<#sources>	<ramin> <ramax> <decmin> <decmax>
        """)
else
    all_boxes = BoundingBox[]
    box_source_counts = Int64[]
    boxes_file = ARGS[1]
    f = open(boxes_file)
    for ln in eachline(f)
        lp = split(ln, '\t')
        if length(lp) != 4
            println("malformed line in box file, skipping remainder")
            println("> $ln")
            break
        end
        sc = parse(Int64, lp[3])
        push!(box_source_counts, sc)
        ss = split(lp[4], ' ')
        ramin = parse(Float64, ss[1])
        ramax = parse(Float64, ss[2])
        decmin = parse(Float64, ss[3])
        decmax = parse(Float64, ss[4])
        bb = BoundingBox(ramin, ramax, decmin, decmax)
        push!(all_boxes, bb)
    end
    close(f)
    outdir = ARGS[2]
    if length(all_boxes) < 1
        println("box file is empty?")
        exit(-1)
    end
    infer_boxes(all_boxes, box_source_counts, ENV["CELESTE_STAGE_DIR"], outdir)
end

