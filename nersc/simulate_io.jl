#!/usr/bin/env julia

using Celeste
using Celeste: SDSSIO
using Gasp
using Gasp: grank
using Base.Threads

const FlatImage = Celeste.SDSSIO.FlatImage

function run_io_simulation(args::Vector{String})
    if length(args) != 2
        println("""
            Usage:
              simulate_io.jl <mode> <boxes_file>
            """)
        exit(-1)
    end
    if !haskey(ENV, "CELESTE_STAGE_DIR")
        Celeste.Log.one_message("ERROR: set CELESTE_STAGE_DIR!")
        exit(-2)
    end
    stagedir = ENV["CELESTE_STAGE_DIR"]

    mode = args[1]
    if !(mode in ["planb","jlds"])
        Celeste.Log.one_message("ERROR: Unsupported mode ($mode), use planb or jlds!")
        exit(-2)
    end
    mode = Symbol(mode)

    # parse the specified box file
    boxfile = args[2]
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
    Celeste.Log.info("dtree: initial: $(ni) ($(ci) to $(li))")
    l = SpinLock()
    iol = SpinLock()
    function datadir(rcf)
        dir = joinpath(stagedir,"mdt$(rcf.run%5)")
        if mode == :planb
            dir = joinpath(dir,"plan_b")
        end
        dir
    end
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
                if mode == :planb
                    this_cat = SDSSIO.read_photoobj_files(rcfs, datadir,
                        duplicate_policy = :primary, slurp = true, drop_quickly = true)
                    images = SDSSIO.load_field_images(rcfs, datadir, true, true)
                else
                    @assert mode == :jlds
                    SDSSIO.load_images_from_jld(rcfs, datadir, false, true)
                end
                bytes, mb = Base.prettyprint_getunits(Sys.maxrss(), length(Base._mem_units), Int64(1024))
                lock(iol)
                Celeste.Log.message("Loaded box $box_idx ($(box.ramin) $(box.ramax) $(box.decmin) $(box.decmax)) with $(length(rcfs)) RCFs in $(toq())s, maxrss is $bytes $(Base._mem_units[mb])")
                unlock(iol)
                # Force full GC to drop all the memory we allocated above
                gc()
            end
        catch exc
            Celeste.Log.exception(exc)
        end
    end
    ccall(:jl_threading_run, Void, (Any,), Core.svec(do_work))
end
run_io_simulation(ARGS)
