#!/usr/bin/env julia

using DataFrames

import Celeste.AccuracyBenchmark
import Celeste.ArgumentParse

parser = Celeste.ArgumentParse.ArgumentParser()
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
scores = AccuracyBenchmark.score_uncertainty(truth, predictions)
println(repr(scores))
