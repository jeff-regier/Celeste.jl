#!/usr/bin/env julia

using DataFrames

import Celeste.AccuracyBenchmark
import Celeste.ArgumentParse

const OUTPUT_DIRECTORY = joinpath(splitdir(Base.source_path())[1], "output")
const CELESTE_PRIOR_CATALOG_CSV = joinpath(OUTPUT_DIRECTORY, "celeste_prior_catalog.csv")

GALAXY_ONLY_COLUMNS = [
    :de_vaucouleurs_mixture_weight,
    :minor_major_axis_ratio,
    :half_light_radius_px,
    :angle_deg,
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
    AccuracyBenchmark.write_catalog(output_filename, catalog)
    AccuracyBenchmark.append_hash_to_file(output_filename)
end

main()
