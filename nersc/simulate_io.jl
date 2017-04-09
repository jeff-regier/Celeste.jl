#!/usr/bin/env julia

using Celeste
using Celeste: SDSSIO
using Gasp

function run_io_simulation(args::Vector{String})
    if length(args) != 1
        println("""
            Usage:
              simulate_io.jl <boxes_file>
            """)
        exit(-1)
    end
    if !haskey(ENV, "CELESTE_STAGE_DIR")
        Celeste.Log.one_message("ERROR: set CELESTE_STAGE_DIR!")
        exit(-2)
    end
    stagedir = ENV["CELESTE_STAGE_DIR"]

    # parse the specified box file
    boxfile = args[1]
    boxes = Vector{BoundingBox}()
    f = open(boxfile)
    for ln in eachline(f)
        lp = split(ln, '\t')
        if length(lp) < 4
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
    end
    close(f)

    field_extents = Celeste.ParallelRun.load_field_extents(stagedir)

    dt, _ = Dtree(length(boxes), 0.25)
    ni, (ci, li) = initwork(dt)
    rundt = runtree(dt)
    if rundt
        Celeste.Log.info("dtree: running tree")
        while runtree(dt)
            Gasp.cpu_pause()
            ccall(:jl_gc_safepoint, Void, ()) 
        end 
        return
    end 
    Celeste.Log.info("dtree: initial: $(ni) ($(ci) to $(li))")

    datadir(rcf) = joinpath(stagedir,"mdt$(rcf.run%5)","plan_b")
    while true
        box_idx = 0
        while true
            li == 0 && break
            if ci > li
                ni, (ci, li) = try getwork(dt)
                catch exc
                    Celeste.Log.exception(exc)
                    break
                end
                if li == 0
                    Celeste.Log.info("dtree: out of work")
                else
                    Celeste.Log.info("dtree: $(ni) work items ($(ci) to $(li))")
                end
            else
                box_idx = ci
                ci += 1
                break
            end
        end
        box_idx == 0 && return

        box = boxes[box_idx]
        rcfs = Celeste.ParallelRun.get_overlapping_fields(box, field_extents)
        tic()
        this_cat = SDSSIO.read_photoobj_files(rcfs, datadir,
            duplicate_policy = :primary, slurp = true)
        images = SDSSIO.load_field_images(rcfs, datadir, true)
        Celeste.Log.message("Loaded box $box_idx in $(toq())")
    end
end
run_io_simulation(ARGS)
