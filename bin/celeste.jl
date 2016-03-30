#!/usr/bin/env julia

using DocOpt
using Dtree
import Celeste

const cpu_hz = 2.3e9

# A workitem is of this ra / dec size
const wira = 0.015
const widec = 0.015

const DOC =
"""Run Celeste.

Usage:
  celeste.jl [options] stage-box <ramin> <ramax> <decmin> <decmax>
  celeste.jl [options] infer-box <ramin> <ramax> <decmin> <decmax> <outdir>
  celeste.jl [options] infer-field <run> <camcol> <field> <outdir>
  celeste.jl [options] infer-object <run> <camcol> <field> <objid> <outdir>
  celeste.jl [options] score-field <run> <camcol> <field> <resultdir> <truthfile>
  celeste.jl [options] score-object <run> <camcol> <field> <objid> <resultdir> <truthfile>
  celeste.jl -h | --help
  celeste.jl --version

Options:
  -h, --help         Show this screen.
  --version          Show the version.
  --logging=<LEVEL>  Level for the Logging package (OFF, DEBUG, INFO, WARNING, ERROR, or CRITICAL). [default: WARNING]
  --stage            In `infer-box`, automatically stage files. Default: false.

The `stage-box` subcommand copies and/or uncompresses all files covering the
given RA/Dec range from /project to user's SCRATCH directory.

The `infer-box` subcommand runs Celeste on all sources in the given
RA/Dec range, using all available images. It depends on `stage-box` having
already been run on the same patch of sky.

The `infer-field` subcommand runs Celeste on all sources in the given
field, using only that field.

The `score-field` subcommand is not yet implemented for the new API.
"""

"""
An area of the sky subtended by `ramin`, `ramax`, `decmin`, and `decmax`.
"""
type SkyArea
    ramin::Float64
    ramax::Float64
    decmin::Float64
    decmax::Float64
    nra::Int64
    ndec::Int64
end

"""
Given a SkyArea that is to be divided into `skya.nra` x `skya.ndec` patches,
return the `i`th patch. `i` is a linear index between 1 and
`skya.nra * skya.ndec`.

This function assumes a cartesian (rather than spherical) coordinate system!
"""
function sky_subarea(skya::SkyArea, i)
    global wira, widec
    ix, iy = ind2sub((skya.nra, skya.ndec), i)

    return (skya.ramin + (ix - 1) * wira,
            min(skya.ramin + ix * wira, skya.ramax),
            skya.decmin + (iy - 1) * widec,
            min(skya.decmin + iy * widec, skya.decmax))
end


"""
Divide the area of sky subtended by `ramin`, `ramax`, `decmin`, and `decmax`
into patches of `wira`X`widec`. Each such patch is a work item. Use Dtree to
distribute these to nodes. Within the node, run inference on each patch
received from Dtree.
"""
function celeste_skyarea(ramin, ramax, decmin, decmax, outdir;
                         stage::Bool=false)
    # ---
    if dt_nodeid == 1
        println("celeste -- running on $dt_nnodes nodes (system clock speed is $cpu_hz Hz)")
        println("  sky area: $(ramax - ramin) X $(decmax - decmin)")
    end

    n_infers = 0
    n_fits = 0
    tm_infer = tm_fldids = tm_rdprim = tm_filcat = tm_rdimg = tm_imp = tm_psf = tm_fit = tm_out = tm_done = 0.0

    if dt_nnodes == 1 # single node run
        tic()
        tm_fldids, tm_rdprim, tm_filcat, tm_rdimg, tm_imp, tm_psf, tm_fit, n_fits, tm_out =
            Celeste.infer_box_nersc(ramin, ramax, decmin, decmax, outdir;
                                    stage=stage)
        tm_infer = toq()
        n_infers = 1

    else # distributed run

        # partition the sky area into `wira` X `widec` sized patches
        global wira, widec
        nra = ceil(Int64, (ramax - ramin) / wira)
        ndec = ceil(Int64, (decmax - decmin) / widec)
        skya = SkyArea(ramin, ramax, decmin, decmax, nra, ndec)

        num_work_items = nra * ndec
        each = ceil(Int64, num_work_items / dt_nnodes)

        if dt_nodeid == 1
            println("  work item dimensions: $wira X $widec")
            println("  $num_work_items work items, ~$each per node")
        end

        # create Dtree and get the initial allocation
        dt = DtreeScheduler(num_work_items, 0.4)
        ni, (ci, li) = initwork(dt)
        rundt = runtree(dt) > 0
        function rundtree()
            if rundt
                rundt = runtree(dt) > 0
                Dtree.cpu_pause()
            end
        end

        # work item processing loop
        Celeste.nputs(dt_nodeid, "initially $ni work items ($ci to $li)")
        tic()
        while ni > 0
            li == 0 && break
            if ci > li
                Celeste.nputs(dt_nodeid, "consumed allocation (last was $li)")
                ni, (ci, li) = getwork(dt)
                Celeste.nputs(dt_nodeid, "got $ni work items ($ci to $li)")
                continue
            end
            item = ci
            ci = ci + 1

            # map item to subarea
            iramin, iramax, idecmin, idecmax = sky_subarea(skya, item)

            # run inference for this subarea
            Celeste.nputs(dt_nodeid, "running inference for $iramin, $iramax, $idecmin, $idecmax")
            tic()
            tm_fi, tm_rp, tm_fc, tm_ri, tm_i, tm_p, tm_f, n_f, tm_o =
                Celeste.infer_box_nersc(iramin, iramax, idecmin, idecmax, outdir;
                                        nid=dt_nodeid, nnodes=dt_nnodes)
            tm_infer = tm_infer + toq()
            n_infers = n_infers + 1
            tm_fldids = tm_fldids + tm_fi
            tm_rdprim = tm_rdprim + tm_rp
            tm_filcat = tm_filcat + tm_fc
            tm_rdimg = tm_rdimg + tm_ri
            tm_imp = tm_imp + tm_i
            tm_psf = tm_psf + tm_p
            tm_fit = tm_fit + tm_f
            n_fits = n_fits + n_f
            tm_out = tm_out + tm_o

            rundtree()
        end
        tm_work = toq()
        Celeste.nputs(dt_nodeid, "out of work")
        tic()
        while rundt
            rundtree()
        end
        finalize(dt)
        tm_done = toq()
    end

    Celeste.nputs(dt_nodeid, "timing ($n_infers sky areas):")
    Celeste.nputs(dt_nodeid, "infer avg: $(tm_infer/n_infers)")
    Celeste.nputs(dt_nodeid, "field ids avg: $(tm_fldids/n_infers)")
    Celeste.nputs(dt_nodeid, "dump output avg: $(tm_out/n_infers)")
    Celeste.nputs(dt_nodeid, "read primary avg: $(tm_rdprim/n_infers)")
    Celeste.nputs(dt_nodeid, "filter catalog avg: $(tm_filcat/n_infers)")
    Celeste.nputs(dt_nodeid, "read images avg: $(tm_rdimg/n_infers)")
    Celeste.nputs(dt_nodeid, "init model params avg: $(tm_imp/n_infers)")
    Celeste.nputs(dt_nodeid, "psf avg: $(tm_psf/n_infers)")
    Celeste.nputs(dt_nodeid, "fit: $((tm_fit/n_fits)/n_infers) (for $n_fits fits)")
    Celeste.nputs(dt_nodeid, "wait for done: $tm_done secs")
end


function time_puts(elapsedtime, bytes, gctime, allocs)
    s = @sprintf("%10.6f seconds", elapsedtime/1e9)
    if bytes != 0 || allocs != 0
        bytes, mb = Base.prettyprint_getunits(bytes, length(Base._mem_units), Int64(1024))
        allocs, ma = Base.prettyprint_getunits(allocs, length(Base._cnt_units), Int64(1000))
        if ma == 1
            s = string(s, @sprintf(" (%d%s allocation%s: ", allocs, Base._cnt_units[ma], allocs==1 ? "" : "s"))
        else
            s = string(s, @sprintf(" (%.2f%s allocations: ", allocs, Base._cnt_units[ma]))
        end
        if mb == 1
            s = string(s, @sprintf("%d %s%s", bytes, Base._mem_units[mb], bytes==1 ? "" : "s"))
        else
            s = string(s, @sprintf("%.3f %s", bytes, Base._mem_units[mb]))
        end
        if gctime > 0
            s = string(s, @sprintf(", %.2f%% gc time", 100*gctime/elapsedtime))
        end
        s = string(s, ")")
    elseif gctime > 0
        s = string(s, @sprintf(", %.2f%% gc time", 100*gctime/elapsedtime))
    end
    Celeste.nputs(dt_nodeid, s)
end


function main()
    Celeste.set_logging_level("WARNING")
    args = docopt(DOC, version=v"0.1.0", options_first=true)
    Celeste.set_logging_level(args["--logging"])
    if args["stage-box"] || args["infer-box"]
        ramin = parse(Float64, args["<ramin>"])
        ramax = parse(Float64, args["<ramax>"])
        decmin = parse(Float64, args["<decmin>"])
        decmax = parse(Float64, args["<decmax>"])
        if args["stage-box"]
            Celeste.stage_box_nersc(ramin, ramax, decmin, decmax)
        elseif args["infer-box"]
            outdir = args["<outdir>"]
            local stats = Base.gc_num()
            local elapsedtime = time_ns()
            celeste_skyarea(ramin, ramax, decmin, decmax, outdir, stage=args["--stage"])
            elapsedtime = time_ns() - elapsedtime
            local diff = Base.GC_Diff(Base.gc_num(), stats)
            time_puts(elapsedtime, diff.allocd, diff.total_time, Base.gc_alloc_count(diff))
        end
    elseif args["infer-field"]
        run = parse(Int, args["<run>"])
        camcol = parse(Int, args["<camcol>"])
        field = parse(Int, args["<field>"])
        if args["infer-field"]
            outdir = args["<outdir>"]
            Celeste.infer_field_nersc(run, camcol, field, outdir)
        elseif args["infer-object"]
            outdir = args["<outdir>"]
            objid = args["<objid>"]
            Celeste.infer_field_nersc(run, camcol, field, outdir; objid=objid)
        elseif args["score-field"]
            resultdir = args["<resultdir>"]
            truthfile = args["<truthfile>"]
            Celeste.score_field_nersc(run, camcol, field, resultdir, truthfile)
        elseif args["score-object"]
            objid = args["<objid>"]
            resultdir = args["<resultdir>"]
            truthfile = args["<truthfile>"]
            Celeste.score_object_nersc(run, camcol, field, objid, resultdir, truthfile)
        end
    end
end


main()
