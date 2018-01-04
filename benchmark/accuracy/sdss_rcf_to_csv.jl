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

rcf = SDSSIO.RunCamcolField(
    parsed_args["run"],
    parsed_args["camcol"],
    parsed_args["field"],
)
@printf("Reading %s...\n", rcf)
catalog_df = AccuracyBenchmark.load_primary(rcf, AccuracyBenchmark.SDSS_DATA_DIR)
@printf("Loaded %d objects from catalog\n", size(catalog_df, 1))

output_filename = @sprintf(
    "sdss_%s_%s_%s_primary.csv",
    rcf.run,
    rcf.camcol,
    rcf.field,
)
output_path = joinpath(OUTPUT_DIRECTORY, output_filename)
output_path = AccuracyBenchmark.write_catalog(output_path, catalog_df; append_hash=true)
@printf("Wrote '%s'\n", output_path)
