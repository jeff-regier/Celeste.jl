"""
    stage_field(run, camcol, field)

Stage all relevant files for the given run, camcol, field to user's SCRATCH
directory. The target locations are given by `field_scratchdir` and
`photofield_scratchdir`.
"""
function stage_field(rcf::RunCamcolField, sdssdir::String, stagedir::String)
    run, camcol, field = rcf.run, rcf.camcol, rcf.field

    # destination directory for all files except photofield.
    camcol_dstdir = joinpath(stagedir, "$run/$camcol")
    field_dstdir = joinpath(stagedir, "$run/$camcol/$field")
    isdir(field_dstdir) || mkpath(field_dstdir)

    # frame files: uncompress bz2 files
    src_frame_dir = "$sdssdir/photoObj/frames/301/$(run)/$(camcol)"
    for band in ['u', 'g', 'r', 'i', 'z']
        srcfile = @sprintf("%s/frame-%s-%06d-%d-%04d.fits.bz2",
                           src_frame_dir, band, run, camcol, field)
        dstfile = @sprintf("%s/frame-%s-%06d-%d-%04d.fits",
                           field_dstdir, band, run, camcol, field)
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
    src_fpm_dir = "$sdssdir/photo/redux/301/$(run)/objcs/$(camcol)"
    for band in ['u', 'g', 'r', 'i', 'z']
        srcfile = @sprintf("%s/fpM-%06d-%s%d-%04d.fit.gz",
                           src_fpm_dir, run, band, camcol, field)
        dstfile = @sprintf("%s/fpM-%06d-%s%d-%04d.fit",
                           field_dstdir, run, band, camcol, field)
        if !isfile(dstfile)
            Log.info("gunzip --stdout $srcfile > $dstfile")
            Base.run(pipeline(`gunzip --stdout $srcfile`, stdout=dstfile))
        end
    end

    # photoobj: simply copy
    src_photoobj_dir = "$sdssdir/photoObj/301/$(run)/$(camcol)"
    srcfile = @sprintf("%s/photoObj-%06d-%d-%04d.fits",
                       src_photoobj_dir, run, camcol, field)
    dstfile = @sprintf("%s/photoObj-%06d-%d-%04d.fits",
                       field_dstdir, run, camcol, field)
    isfile(dstfile) || cp(srcfile, dstfile)

    # psField: simply copy
    src_psfield_dir = "$sdssdir/photo/redux/301/$(run)/objcs/$(camcol)"
    srcfile = @sprintf("%s/psField-%06d-%d-%04d.fit",
                       src_psfield_dir, run, camcol, field)
    dstfile = @sprintf("%s/psField-%06d-%d-%04d.fit",
                       field_dstdir, run, camcol, field)
    isfile(dstfile) || cp(srcfile, dstfile)

    # photofield: simply copy
    src_photofield_dir = "$sdssdir/photoObj/301/$(run)"
    srcfile = @sprintf("%s/photoField-%06d-%d.fits",
                       src_photofield_dir, run, camcol)
    dstfile = @sprintf("%s/photoField-%06d-%d.fits",
                       camcol_dstdir, run, camcol)
    isfile(dstfile) || cp(srcfile, dstfile)
end


"""
Stage all relevant files for the given sky patch to user's SCRATCH.
"""
function stage_box(box::BoundingBox, sdssdir, stagedir)
    mkpath(stagedir)
    stage_extents = "$stagedir/field_extents.fits"
    cp(ENV["FIELD_EXTENTS"], stage_extents, remove_destination=true)
    rcfs = get_overlapping_fields(box, stagedir)
    for rcf in rcfs
        println("$(rcf.run) $(rcf.camcol) $(rcf.field)")
    end
end

