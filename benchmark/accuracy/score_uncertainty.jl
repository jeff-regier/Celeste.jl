#!/usr/bin/env julia

using DataFrames

import Celeste.AccuracyBenchmark
import Celeste.ArgumentParse

parser = ArgumentParse.ArgumentParser()
ArgumentParse.add_argument(
    parser,
    "--write-error-csv",
    help="Write raw errors to the given CSV file",
)
ArgumentParse.add_argument(
    parser,
    "ground_truth_csv",
    help="Ground truth CSV catalog",
)
ArgumentParse.add_argument(
    parser,
    "celeste_prediction_csv",
    help="Celeste predictions CSV catalog",
)
parsed_args = ArgumentParse.parse_args(parser, ARGS)

truth = AccuracyBenchmark.read_catalog(parsed_args["ground_truth_csv"])
predictions = AccuracyBenchmark.read_catalog(parsed_args["celeste_prediction_csv"])
uncertainty_df = AccuracyBenchmark.get_uncertainty_df(truth, predictions)
scores = AccuracyBenchmark.score_uncertainty(uncertainty_df)
println(repr(scores))

if haskey(parsed_args, "write-error-csv")
    writetable(parsed_args["write-error-csv"], uncertainty_df)
end
