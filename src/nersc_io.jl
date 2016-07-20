# NERSC-specific functions

# TODO: make a config file for NERSC-specific paths, and make everything else
# in here not specific to NERSC


# NERSC scratch directories
nersc_field_scratchdir(run::Integer, camcol::Integer, field::Integer) =
    joinpath(ENV["SCRATCH"], "celeste/$(run)/$(camcol)/$(field)")
nersc_photofield_scratchdir(run::Integer, camcol::Integer) =
    joinpath(ENV["SCRATCH"], "celeste/$(run)/$(camcol)")

"""
    nersc_stage_field(run, camcol, field)

Stage all relevant files for the given run, camcol, field to user's SCRATCH
directory. The target locations are given by `nersc_field_scratchdir` and
`nersc_photofield_scratchdir`.
"""
function nersc_stage_field(run::Integer, camcol::Integer, field::Integer)
    # destination directory for all files except photofield.
    dstdir = nersc_field_scratchdir(run, camcol, field)
    isdir(dstdir) || mkpath(dstdir)

    # frame files: uncompress bz2 files
    srcdir = nersc_frame_dir(run, camcol)
    for band in ['u', 'g', 'r', 'i', 'z']
        srcfile = @sprintf("%s/frame-%s-%06d-%d-%04d.fits.bz2",
                           srcdir, band, run, camcol, field)
        dstfile = @sprintf("%s/frame-%s-%06d-%d-%04d.fits",
                           dstdir, band, run, camcol, field)
        if !isfile(dstfile)
            Lumberjack.info("bzcat --keep $srcfile > $dstfile")
            Base.run(pipeline(`bzcat --keep $srcfile`, stdout=dstfile))
        end
    end

    # fpm files
    # It isn't strictly necessary to uncompress these, because FITSIO can handle
    # gzipped files. However, the celeste code assumes the filename ends with
    # ".fit", so we would have to at least change the name. It seems clearer
    # to simply uncompress here.
    srcdir = nersc_fpm_dir(run, camcol)
    dstdir = nersc_field_scratchdir(run, camcol, field)
    isdir(dstdir) || mkpath(dstdir)
    for band in ['u', 'g', 'r', 'i', 'z']
        srcfile = @sprintf("%s/fpM-%06d-%s%d-%04d.fit.gz",
                           srcdir, run, band, camcol, field)
        dstfile = @sprintf("%s/fpM-%06d-%s%d-%04d.fit",
                           dstdir, run, band, camcol, field)
        if !isfile(dstfile)
            Lumberjack.info("gunzip --stdout $srcfile > $dstfile")
            Base.run(pipeline(`gunzip --stdout $srcfile`, stdout=dstfile))
        end
    end

    # photoobj: simply copy
    srcfile = @sprintf("%s/photoObj-%06d-%d-%04d.fits",
                       nersc_photoobj_dir(run, camcol), run, camcol, field)
    dstfile = @sprintf("%s/photoObj-%06d-%d-%04d.fits",
                       nersc_field_scratchdir(run, camcol, field), run,
                       camcol, field)
    isfile(dstfile) || cp(srcfile, dstfile)

    # psField: simply copy
    srcfile = @sprintf("%s/psField-%06d-%d-%04d.fit",
                       nersc_psfield_dir(run, camcol), run, camcol, field)
    dstfile = @sprintf("%s/psField-%06d-%d-%04d.fit",
                       nersc_field_scratchdir(run, camcol, field), run,
                       camcol, field)
    isfile(dstfile) || cp(srcfile, dstfile)

    # photofield: simply copy
    srcfile = @sprintf("%s/photoField-%06d-%d.fits",
                       nersc_photofield_dir(run), run, camcol)
    dstfile = @sprintf("%s/photoField-%06d-%d.fits",
                       nersc_photofield_scratchdir(run, camcol), run, camcol)
    isfile(dstfile) || cp(srcfile, dstfile)
end


"""
Stage all relevant files for the given sky patch to user's SCRATCH.
"""
function stage_box_nersc(ramin, ramax, decmin, decmax)
    fieldids = query_overlapping_fieldids(ramin, ramax, decmin, decmax)
    for (run, camcol, field) in fieldids
        nersc_stage_field(run, camcol, field)
    end
end


"""
NERSC-specific infer function, called from main entry point.
"""
function infer_box_nersc(ramin, ramax, decmin, decmax, outdir)
    # Base.@time hack for distributed environment
    gc_stats = ()
    gc_diff_stats = ()
    elapsed_time = 0.0
    gc_stats = Base.gc_num()
    elapsed_time = time_ns()

    times = InferTiming()
    if dt_nnodes > 1
        divide_and_infer((ramin, ramax),
                         (decmin, decmax),
                         timing=times,
                         outdir=outdir)
    else
        # Get vector of (run, camcol, field) triplets overlapping this patch
        tic()
        fieldids = query_overlapping_fieldids(ramin, ramax,
                                              decmin, decmax)
        times.query_fids = toq()

        # Get relevant directories corresponding to each field.
        tic()
        frame_dirs = query_frame_dirs(fieldids)
        photofield_dirs = query_photofield_dirs(fieldids)
        times.get_dirs = toq()

        results = infer(fieldids,
                        frame_dirs;
                        ra_range=(ramin, ramax),
                        dec_range=(decmin, decmax),
                        fpm_dirs=frame_dirs,
                        psfield_dirs=frame_dirs,
                        photoobj_dirs=frame_dirs,
                        photofield_dirs=photofield_dirs,
                        timing=times)

        tic()
        save_results(outdir, ramin, ramax, decmin, decmax, results)
        times.write_results = toq()
    end

    # Base.@time hack for distributed environment
    elapsed_time = time_ns() - elapsed_time
    gc_diff_stats = Base.GC_Diff(Base.gc_num(), gc_stats)
    time_puts(elapsed_time, gc_diff_stats.allocd, gc_diff_stats.total_time,
              Base.gc_alloc_count(gc_diff_stats))

    times.num_srcs = max(1, times.num_srcs)
    nputs(dt_nodeid, "timing: query_fids=$(times.query_fids)")
    nputs(dt_nodeid, "timing: get_dirs=$(times.get_dirs)")
    nputs(dt_nodeid, "timing: num_infers=$(times.num_infers)")
    nputs(dt_nodeid, "timing: read_photoobj=$(times.read_photoobj)")
    nputs(dt_nodeid, "timing: read_img=$(times.read_img)")
    nputs(dt_nodeid, "timing: init_mp=$(times.init_mp)")
    nputs(dt_nodeid, "timing: fit_psf=$(times.fit_psf)")
    nputs(dt_nodeid, "timing: opt_srcs=$(times.opt_srcs)")
    nputs(dt_nodeid, "timing: num_srcs=$(times.num_srcs)")
    nputs(dt_nodeid, "timing: average opt_srcs=$(times.opt_srcs/times.num_srcs)")
    nputs(dt_nodeid, "timing: write_results=$(times.write_results)")
    nputs(dt_nodeid, "timing: wait_done=$(times.wait_done)")
end


"""
Query the SDSS database for all fields that overlap the given RA, Dec range.
"""
function query_overlapping_fields(ramin, ramax, decmin, decmax)

    fname = "/project/projectdirs/dasrepo/celeste-sc16/field_extents.fits"

    f = FITSIO.FITS(fname)
    hdu = f[2]::FITSIO.TableHDU

    # read in the entire table.
    all_run = read(hdu, "run")::Vector{Int16}
    all_camcol = read(hdu, "camcol")::Vector{UInt8}
    all_field = read(hdu, "field")::Vector{Int16}
    all_ramin = read(hdu, "ramin")::Vector{Float64}
    all_ramax = read(hdu, "ramax")::Vector{Float64}
    all_decmin = read(hdu, "decmin")::Vector{Float64}
    all_decmax = read(hdu, "decmax")::Vector{Float64}

    close(f)

    # initialize output "table"
    out = Dict{String, Vector}("run"=>Int[],
                                    "camcol"=>Int[],
                                    "field"=>Int[],
                                    "ramin"=>Float64[],
                                    "ramax"=>Float64[],
                                    "decmin"=>Float64[],
                                    "decmax"=>Float64[])

    # The ramin, ramax, etc is a bit unintuitive because we're looking
    # for any overlap.
    for i in eachindex(all_ramin)
        if (all_ramax[i] > ramin && all_ramin[i] < ramax &&
            all_decmax[i] > decmin && all_decmin[i] < decmax)
            push!(out["run"], all_run[i])
            push!(out["camcol"], all_camcol[i])
            push!(out["field"], all_field[i])
            push!(out["ramin"], all_ramin[i])
            push!(out["ramax"], all_ramax[i])
            push!(out["decmin"], all_decmin[i])
            push!(out["decmax"], all_decmax[i])
        end
    end

    return out
end

"""
query_overlapping_fieldids(ramin, ramax, decmin, decmax) -> Vector{Tuple{Int, Int, Int}}

Like `query_overlapping_fields`, but return a Vector of
(run, camcol, field) triplets.
"""
function query_overlapping_fieldids(ramin, ramax, decmin, decmax)
    fields = query_overlapping_fields(ramin, ramax, decmin, decmax)
    return Tuple{Int, Int, Int}[(fields["run"][i],
                                 fields["camcol"][i],
                                 fields["field"][i])
                                for i in eachindex(fields["run"])]
end

query_frame_dirs(fieldids) =
    [nersc_field_scratchdir(x[1], x[2], x[3]) for x in fieldids]
query_photofield_dirs(fieldids) =
    [nersc_photofield_scratchdir(x[1], x[2]) for x in fieldids]

# NERSC source directories
const NERSC_DATA_ROOT = "/global/projecta/projectdirs/sdss/data/sdss/dr12/boss"
nersc_photoobj_dir(run::Integer, camcol::Integer) =
    "$(NERSC_DATA_ROOT)/photoObj/301/$(run)/$(camcol)"
nersc_psfield_dir(run::Integer, camcol::Integer) =
    "$(NERSC_DATA_ROOT)/photo/redux/301/$(run)/objcs/$(camcol)"
nersc_photofield_dir(run::Integer) =
    "$(NERSC_DATA_ROOT)/photoObj/301/$(run)"
nersc_frame_dir(run::Integer, camcol::Integer) =
    "$(NERSC_DATA_ROOT)/photoObj/frames/301/$(run)/$(camcol)"
nersc_fpm_dir(run::Integer, camcol::Integer) =
    "$(NERSC_DATA_ROOT)/photo/redux/301/$(run)/objcs/$(camcol)"


"""
NERSC-specific infer function, called from main entry point.
"""
function infer_field_nersc(run::Int, camcol::Int, field::Int,
                           outdir::AbstractString; objid="")
    # ensure that files are staged and set up paths.
    nersc_stage_field(run, camcol, field)
    field_dirs = [nersc_field_scratchdir(run, camcol, field)]
    photofield_dirs = [nersc_photofield_scratchdir(run, camcol)]

    results = infer([(run, camcol, field)], field_dirs;
                    objid=objid,
                    fpm_dirs=field_dirs,
                    psfield_dirs=field_dirs,
                    photoobj_dirs=field_dirs,
                    photofield_dirs=photofield_dirs,
                    primary_initialization=false)

    fname = if objid == ""
        @sprintf "%s/celeste-%06d-%d-%04d.jld" outdir run camcol field
    else
        @sprintf "%s/celeste-objid-%s.jld" outdir objid
    end
    JLD.save(fname, "results", results)
    Lumberjack.debug("infer_field_nersc finished successfully")
end


"""
Save provided results to a JLD file.
"""
function save_results(outdir, ramin, ramax, decmin, decmax, results)
    fname = @sprintf("%s/celeste-%.4f-%.4f-%.4f-%.4f.jld",
                     outdir, ramin, ramax, decmin, decmax)
    JLD.save(fname, "results", results)
end


