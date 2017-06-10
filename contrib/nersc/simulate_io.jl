#!/usr/bin/env julia

using Celeste
using Celeste: SDSSIO
using Gasp
using Gasp: grank
using Base.Threads

include("../bin/binutil.jl")

function run_io_simulation(args::Vector{String})
    if length(args) != 3
        println("""
            Usage:
              simulate_io.jl <strategy> <rcf_nsrcs_file> <boxes_file>
            """)
        exit(-1)
    end
    strategy, all_rcfs, all_rcf_nsrcs = decide_strategy(args[1], args[2])

    # parse the specified box file
    boxfile = args[3]
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

    field_extents = Celeste.ParallelRun.load_field_extents(strategy)

    dt, _ = Dtree(length(boxes), 0.25)
    ni, (ci, li) = initwork(dt)
    rundt = runtree(dt)
    Celeste.Log.info("dtree: initial: $(ni) ($(ci) to $(li))")
    l = SpinLock()
    iol = SpinLock()
    function do_work()
        try
            if rundt && threadid() == nthreads()
                Celeste.Log.info("dtree: running tree")
                while runtree(dt)
                    Gasp.cpu_pause()
                    ccall(:jl_gc_safepoint, Void, ())
                end
                return
            end

            while true
                box_idx = 0
                lock(l)
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
                unlock(l)
                box_idx == 0 && return

                box = boxes[box_idx]
                rcfs = Celeste.ParallelRun.get_overlapping_fields(box, field_extents)
                tic()
                for rcf in rcfs
                    # Do this one RCF at a time, we don't have the memory to keep all images
                    states = SDSSIO.preload_rcfs(strategy, [rcf])
                    SDSSIO.read_photoobj(strategy, rcf, states[], drop_quickly = true)
                    SDSSIO.load_field_images(strategy, [rcf], states, #= drop_quickly = =# true)
                end
                bytes, mb = Base.prettyprint_getunits(Sys.maxrss(), length(Base._mem_units), Int64(1024))
                lock(iol)
                Celeste.Log.message("Loaded box $box_idx ($(box.ramin) $(box.ramax) $(box.decmin) $(box.decmax)) with $(length(rcfs)) RCFs in $(toq())s, maxrss is $bytes $(Base._mem_units[mb])")
                unlock(iol)
                # Force full GC to drop all the memory we allocated above
            end
        catch exc
            Celeste.Log.exception(exc)
        end
    end
    ccall(:jl_threading_run, Void, (Any,), Core.svec(do_work))
end
run_io_simulation(ARGS)
