#!/usr/bin/env julia

using DataFrames

import Celeste.AccuracyBenchmark

const OUTPUT_DIRECTORY = joinpath(splitdir(Base.source_path())[1], "output")
const COADD_CATALOG_CSV = joinpath(OUTPUT_DIRECTORY, "stripe82_coadd_catalog.csv")
const CELESTE_PRIOR_CATALOG_CSV = joinpath(OUTPUT_DIRECTORY, "celeste_prior_catalog.csv")

GALAXY_ONLY_COLUMNS = [
    :de_vaucouleurs_mixture_weight,
    :minor_major_axis_ratio,
    :half_light_radius_px,
    :angle_deg,
]

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
const COADD_CATALOG_FITS = joinpath(Pkg.dir("Celeste"), "test", "data", "coadd_for_4263_5_119.fit")

function print_usage()
    println("Generate a catalog in CSV format.")
    println("Usage: write_ground_truth_catalog_csv.jl (coadd | prior)")
    println("  coadd: convert the Stripe82 coadd catalog to CSV form.")
    println("  prior: draw sources at random from the Celeste prior.")
end

function main()
    if length(ARGS) != 1
        println("Missing action argument.\n")
        print_usage()
        return
    end

    action = ARGS[1]
    if action == "coadd"
        catalog = AccuracyBenchmark.load_coadd_catalog(COADD_CATALOG_FITS)
        output_filename = COADD_CATALOG_CSV
    elseif action == "prior"
        catalog = AccuracyBenchmark.generate_catalog_from_celeste_prior(500, 12345)
        output_filename = CELESTE_PRIOR_CATALOG_CSV
    else
        @printf("Invalid action: '%s'\n\n", action)
        print_usage()
        return
    end

    # for stars, ensure galaxy-only fields are NA
    for column_symbol in GALAXY_ONLY_COLUMNS
        catalog[catalog[:is_star], column_symbol] = NA
    end

    if !isdir(OUTPUT_DIRECTORY)
        mkdir(OUTPUT_DIRECTORY)
    end
    AccuracyBenchmark.write_catalog(output_filename, catalog)
    AccuracyBenchmark.append_hash_to_file(output_filename)
end

main()
