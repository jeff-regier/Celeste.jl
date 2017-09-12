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
parsed_args = ArgumentParse.parse_args(parser, ARGS)

run_camcol_field = SDSSIO.RunCamcolField(
    parsed_args["run"],
    parsed_args["camcol"],
    parsed_args["field"],
)
@printf("Reading %s...\n", run_camcol_field)
catalog_df = AccuracyBenchmark.load_primary(run_camcol_field, AccuracyBenchmark.SDSS_DATA_DIR)
@printf("Loaded %d objects from catalog\n", size(catalog_df, 1))

output_filename = @sprintf(
    "sdss_%s_%s_%s_primary.csv",
    run_camcol_field.run,
    run_camcol_field.camcol,
    run_camcol_field.field,
)
output_path = joinpath(OUTPUT_DIRECTORY, output_filename)
AccuracyBenchmark.write_catalog(output_path, catalog_df)
AccuracyBenchmark.append_hash_to_file(output_path)
