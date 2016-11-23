#!/usr/bin/env julia

import Celeste.ParallelRun: BoundingBox, infer_field
import Celeste.Stripe82Score: score_field_disk, score_object_disk
import Celeste.SDSSIO: RunCamcolField


# I'd rather let the user specify a rcf on the command line, but picking
# an arbitrary rcf isn't too useful without having ground truth for it,
# from a co-add run. Getting ground truth isn't easy to automate because
# the web ui is the best way to interact with the CasJobs server.
# At least having these hard coded ensures that we don't compare scores
# from different regions.
const rcf = RunCamcolField(4263, 5, 119)


# This data might already be there from the unit tests.
# `make` won't fetch it again if it's already there.
const datadir = joinpath(Pkg.dir("Celeste"), "test", "data")
wd = pwd()
cd(datadir)
run(`make RUN=$(rcf.run) CAMCOL=$(rcf.camcol) FIELD=$(rcf.field)`)
cd(wd)


# We could also use the current directory. I'd rather not burden
# the user with another directory to specify on the command line,
const outdir = joinpath(Pkg.dir("Celeste"), "benchmark", "stripe82")


# The truth file comes from CasJobs. The query that generates it appears
# below. I selected the RA/Dec range in this query to include all of the
# (4263, 5, 119) rcf, but not a lot else. In the query, `run` is limited
# to 106 and 206 because these are the coadd runs. The nested select
# structure is important for getting this query to complete in time.
"""
#declare @BRIGHT bigint set @BRIGHT=dbo.fPhotoFlags('BRIGHT')
declare @EDGE bigint set @EDGE=dbo.fPhotoFlags('EDGE')
declare @SATURATED bigint set @SATURATED=dbo.fPhotoFlags('SATURATED')
declare @NODEBLEND bigint set @NODEBLEND=dbo.fPhotoFlags('NODEBLEND')
declare @bad_flags bigint set
@bad_flags=(@SATURATED|@BRIGHT|@EDGE|@NODEBLEND)

select *
from (
select
  objid, rerun, run, camcol, field, flags,
  ra, dec, probpsf,
  psfmag_u, psfmag_g, psfmag_r, psfmag_i, psfmag_z,
  devmag_u, devmag_g, devmag_r, devmag_i, devmag_z,
  expmag_u, expmag_g, expmag_r, expmag_i, expmag_z,
  fracdev_r,
  devab_r, expab_r,
  devphi_r, expphi_r,
  devrad_r, exprad_r
into coadd_field_catalog
from stripe82.photoobj
where
  run in (106, 206) and
  ra between 0.449 and 0.599 and
  dec between 0.417 and 0.629) as tmp
where
  ((psfmag_i < 22 and probpsf = 1) or (probpsf = 0 and (expmag_i < 22 or devmag_i < 22))) and
  (flags & @bad_flags) = 0
"""
const truthfile = joinpath(datadir, "coadd_for_4263_5_119.fit")


if length(ARGS) > 1 || (length(ARGS) == 1 && ARGS[1] != "--score-only")
    println("usage: validate_on_stripe82.jl [--score-only]")
else
    # By default this script both infers all the parameters and scores them,
    # but because inference is computationally intensive, whereas scoring isn't,
    # the user gets the option to just run the scoring mode. Running a scoring
    # alone would primarily be useful for debugging.
    if length(ARGS) == 0 || ARGS[1] != "--score-only"
        # adding `; objid=...` to the call limits infer_field to that source.
        # Potentially useful for debugging.
        @time infer_field(rcf, datadir, outdir)
    end

    # Calling `score_object_disk(rcf, objid, datadir, truthfile, datadir)`
    # instead limits scoring to the specific light source (objid).
    # That could be somewhat useful for debugging. The output is in a somewhat
    # different format though, because with just one object it doesn't make
    # sense to compute a full table comparing Celeste to Primary. 
    score_field_disk(rcf, outdir, truthfile, datadir)
end

