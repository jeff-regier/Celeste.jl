import Celeste.SDSSIO: RunCamcolField, PlainFITSStrategy, BigFileStrategy

function decide_strategy(strategyarg, nsrcsarg)
    if !haskey(ENV, "CELESTE_STAGE_DIR")
        Log.one_message("ERROR: set CELESTE_STAGE_DIR!")
        exit(-2)
    end

    # load the RCFs #sources file
    rcf_nsrcs_file = nsrcsarg
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

    strategy = PlainFITSStrategy(ENV["CELESTE_STAGE_DIR"])
    if !isempty(strategyarg)
        which = strategyarg
        if which ==  "mdtfits"
            stagedir = strategy.stagedir
            datadir(rcf) = joinpath(stagedir,"mdt$(rcf.run%5)","plan_b")
            strategy = PlainFITSStrategy(stagedir, datadir, false)
        elseif which == "mdtfitsslurp"
            stagedir = strategy.stagedir
            datadir(rcf) = joinpath(stagedir,"mdt$(rcf.run%5)","plan_b")
            strategy = PlainFITSStrategy(stagedir, datadir, true)
        elseif which == "bigfiles"
            runs = unique(map(rcf->rcf.run, all_rcfs))
            run_idx_map = Dict(r=>i for (i,r) in enumerate(runs))
            rcf_idx_map = Dict(rcf=>i for (i,rcf) in enumerate(all_rcfs))
            bfo = SDSSIO.init_bigfile_io(strategy.stagedir)
            strategy = BigFileStrategy(strategy.stagedir, bfo, rcf_idx_map, run_idx_map)
        else
            Log.one_message("ERROR: Unknown IO strategy $(which)")
            exit(-2)
        end
    end
    strategy, all_rcfs, all_rcf_nsrcs
end
