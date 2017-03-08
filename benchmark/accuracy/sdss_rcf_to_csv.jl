#!/usr/bin/env julia

using DataFrames
import FITSIO

import Celeste.AccuracyBenchmark
import Celeste.ArgumentParse
import Celeste.SDSSIO

const OUTPUT_DIRECTORY = joinpath(splitdir(Base.source_path())[1], "output")

parser = Celeste.ArgumentParse.ArgumentParser()
ArgumentParse.add_argument(
    parser,
    "--run",
    help="SDSS run #",
    arg_type=Int,
    default=AccuracyBenchmark.STRIPE82_RCF.run,
)
ArgumentParse.add_argument(
    parser,
    "--camcol",
    help="SDSS camcol #",
    arg_type=Int,
    default=AccuracyBenchmark.STRIPE82_RCF.camcol,
)
ArgumentParse.add_argument(
    parser,
    "--field",
    help="SDSS field #",
    arg_type=Int,
    default=AccuracyBenchmark.STRIPE82_RCF.field,
)
ArgumentParse.add_argument(
    parser,
    "truth_catalog_csv",
    help = "CSV file containing coadd 'ground truth' catalog",
    required = true,
)
parsed_args = ArgumentParse.parse_args(parser, ARGS)

run_camcol_field = SDSSIO.RunCamcolField(
    parsed_args["run"],
    parsed_args["camcol"],
    parsed_args["field"],
)
@printf("Reading %s...\n", run_camcol_field)
catalog_df = AccuracyBenchmark.load_primary(run_camcol_field, AccuracyBenchmark.SDSS_DATA_DIR)
@printf("Loaded %d objects from catalog\n", size(catalog_df, 1))

# Match sources to ground truth catalog, and pull object IDs from truth catalog
catalog_df[:objid] = fill("", size(catalog_df, 1))
truth_df = AccuracyBenchmark.read_catalog(parsed_args["truth_catalog_csv"])
used_truth_indices = Set()
for source_index in 1:size(catalog_df, 1)
    matching_truth_index = nothing
    try
        matching_truth_index = AccuracyBenchmark.match_position(
            truth_df[:right_ascension_deg], truth_df[:declination_deg],
            catalog_df[source_index, :right_ascension_deg],
            catalog_df[source_index, :declination_deg],
            1,
        )
    catch exc
        if !isa(exc, AccuracyBenchmark.MatchException)
            rethrow()
        else
            continue
        end
    end

    if in(matching_truth_index, used_truth_indices)
        continue
    else
        catalog_df[source_index, :objid] = string(truth_df[matching_truth_index, :objid])
        union!(used_truth_indices, matching_truth_index)
    end
end

catalog_df = catalog_df[catalog_df[:objid] .!= "", :]
@printf("Matched %d objects to ground truth catalog\n", size(catalog_df, 1))

output_filename = @sprintf(
    "sdss_%s_%s_%s_primary.csv",
    run_camcol_field.run,
    run_camcol_field.camcol,
    run_camcol_field.field,
)
output_path = joinpath(OUTPUT_DIRECTORY, output_filename)
AccuracyBenchmark.write_catalog(output_path, catalog_df)
AccuracyBenchmark.append_hash_to_file(output_path)
