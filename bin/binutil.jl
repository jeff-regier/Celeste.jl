import Celeste.SDSSIO: RunCamcolField, PlainFITSStrategy

function parse_rcfs_nsrcs(rcf_nsrcs_file)
    all_rcfs = Vector{RunCamcolField}()
    all_rcf_nsrcs = Vector{Int16}()
    open(rcf_nsrcs_file) do f
        for ln in eachline(f)
            lp = split(ln, '\t')
            run = parse(Int16, lp[1])
            camcol = parse(UInt8, lp[2])
            field = parse(Int16, lp[3])
            nsrc = parse(Int16, lp[4])
            push!(all_rcfs, RunCamcolField(run, camcol, field))
            push!(all_rcf_nsrcs, nsrc)
        end
    end
    all_rcfs, all_rcf_nsrcs
end

function parse_boxfile(boxfile)
    boxes = Vector{BoundingBox}()
    boxes_rcf_idxs = Vector{Vector{Int32}}()
    open(boxfile) do f
        for ln in eachline(f)
            lp = split(ln, '\t')
            if length(lp) != 5
                Log.one_message("ERROR: malformed line in box file:\n> $ln ")
                return boxes, boxes_rcf_idxs
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
    end
    if length(boxes) < 1
        Log.one_message("$boxfile is empty?")
    end
    return boxes, boxes_rcf_idxs
end
