#!/usr/bin/env julia

using DataFrames

import Celeste.AccuracyBenchmark
import Celeste.ArgumentParse

const OUTPUT_DIRECTORY = joinpath(splitdir(Base.source_path())[1], "output")
const CELESTE_PRIOR_CATALOG_CSV = joinpath(OUTPUT_DIRECTORY, "prior.csv")
const COADD_CATALOG_FITS = joinpath(Pkg.dir("Celeste"), "test", "data", "coadd_for_4263_5_119.fit")

GALAXY_ONLY_COLUMNS = [
    :gal_frac_dev,
    :gal_axis_ratio,
    :gal_radius_px,
    :gal_angle_deg,
]

parser = Celeste.ArgumentParse.ArgumentParser()
ArgumentParse.add_argument(
    parser,
    "action",
    help=string(
        "coadd: convert a Stripe82 coadd catalog to CSV form; ",
        "prior: draw sources at random from the Celeste prior",
    ),
)
ArgumentParse.add_argument(
    parser,
    "coadd_fits",
    help="Stripe82 catalog FITS file, for 'coadd'",
    default=COADD_CATALOG_FITS,
    required=false,
)
parsed_args = ArgumentParse.parse_args(parser, ARGS)

function main()
    if parsed_args["action"] == "coadd"
        catalog = AccuracyBenchmark.load_coadd_catalog(parsed_args["coadd_fits"])
        catalog_label = splitext(basename(parsed_args["coadd_fits"]))[1]
        output_filename = joinpath(OUTPUT_DIRECTORY, "$(catalog_label).csv")
    elseif parsed_args["action"] == "prior"
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
    output_filename = AccuracyBenchmark.write_catalog(
        output_filename, catalog_df; append_hash=true)
    @printf("Wrote '%s'\n", output_filename)
end

main()
