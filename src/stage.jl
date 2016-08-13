# source directories---copy from these to scratch
photoobj_dir(run::Integer, camcol::Integer) =
    "$(ENV["SDSS_ROOT_DIR"])/photoObj/301/$(run)/$(camcol)"
psfield_dir(run::Integer, camcol::Integer) =
    "$(ENV["SDSS_ROOT_DIR"])/photo/redux/301/$(run)/objcs/$(camcol)"
photofield_dir(run::Integer) =
    "$(ENV["SDSS_ROOT_DIR"])/photoObj/301/$(run)"
frame_dir(run::Integer, camcol::Integer) =
    "$(ENV["SDSS_ROOT_DIR"])/photoObj/frames/301/$(run)/$(camcol)"
fpm_dir(run::Integer, camcol::Integer) =
    "$(ENV["SDSS_ROOT_DIR"])/photo/redux/301/$(run)/objcs/$(camcol)"

# scratch directories
field_scratchdir(run::Integer, camcol::Integer, field::Integer) =
    joinpath(ENV["SCRATCH"], "celeste/$(run)/$(camcol)/$(field)")
photofield_scratchdir(run::Integer, camcol::Integer) =
    joinpath(ENV["SCRATCH"], "celeste/$(run)/$(camcol)")
extents_scratch() =
    joinpath(ENV["SCRATCH"], "celeste/field_extents.fits")

"""
    stage_field(run, camcol, field)

Stage all relevant files for the given run, camcol, field to user's SCRATCH
directory. The target locations are given by `field_scratchdir` and
`photofield_scratchdir`.
"""
function stage_field(run::Integer, camcol::Integer, field::Integer)
    # destination directory for all files except photofield.
    dstdir = field_scratchdir(run, camcol, field)
    isdir(dstdir) || mkpath(dstdir)

    # frame files: uncompress bz2 files
    srcdir = frame_dir(run, camcol)
    for band in ['u', 'g', 'r', 'i', 'z']
        srcfile = @sprintf("%s/frame-%s-%06d-%d-%04d.fits.bz2",
                           srcdir, band, run, camcol, field)
        dstfile = @sprintf("%s/frame-%s-%06d-%d-%04d.fits",
                           dstdir, band, run, camcol, field)
        if !isfile(dstfile)
            Log.info("bzcat --keep $srcfile > $dstfile")
            Base.run(pipeline(`bzcat --keep $srcfile`, stdout=dstfile))
        end
    end

    # fpm files
    # It isn't strictly necessary to uncompress these, because FITSIO can handle
    # gzipped files. However, the celeste code assumes the filename ends with
    # ".fit", so we would have to at least change the name. It seems clearer
    # to simply uncompress here.
    srcdir = fpm_dir(run, camcol)
    dstdir = field_scratchdir(run, camcol, field)
    isdir(dstdir) || mkpath(dstdir)
    for band in ['u', 'g', 'r', 'i', 'z']
        srcfile = @sprintf("%s/fpM-%06d-%s%d-%04d.fit.gz",
                           srcdir, run, band, camcol, field)
        dstfile = @sprintf("%s/fpM-%06d-%s%d-%04d.fit",
                           dstdir, run, band, camcol, field)
        if !isfile(dstfile)
            Log.info("gunzip --stdout $srcfile > $dstfile")
            Base.run(pipeline(`gunzip --stdout $srcfile`, stdout=dstfile))
        end
    end

    # photoobj: simply copy
    srcfile = @sprintf("%s/photoObj-%06d-%d-%04d.fits",
                       photoobj_dir(run, camcol), run, camcol, field)
    dstfile = @sprintf("%s/photoObj-%06d-%d-%04d.fits",
                       field_scratchdir(run, camcol, field), run,
                       camcol, field)
    isfile(dstfile) || cp(srcfile, dstfile)

    # psField: simply copy
    srcfile = @sprintf("%s/psField-%06d-%d-%04d.fit",
                       psfield_dir(run, camcol), run, camcol, field)
    dstfile = @sprintf("%s/psField-%06d-%d-%04d.fit",
                       field_scratchdir(run, camcol, field), run,
                       camcol, field)
    isfile(dstfile) || cp(srcfile, dstfile)

    # photofield: simply copy
    srcfile = @sprintf("%s/photoField-%06d-%d.fits",
                       photofield_dir(run), run, camcol)
    dstfile = @sprintf("%s/photoField-%06d-%d.fits",
                       photofield_scratchdir(run, camcol), run, camcol)
    isfile(dstfile) || cp(srcfile, dstfile)
end


"""
Stage all relevant files for the given sky patch to user's SCRATCH.
"""
function stage_box(ramin, ramax, decmin, decmax)
    mkpath("$(ENV["SCRATCH"])/celeste")
    cp(ENV["FIELD_EXTENTS"], extents_scratch())
    fieldids = query_overlapping_fieldids(ramin, ramax, decmin, decmax)
    for (run, camcol, field) in fieldids
        stage_field(run, camcol, field)
    end
end

