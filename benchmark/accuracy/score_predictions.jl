#!/usr/bin/env julia

using DataFrames

import Celeste.AccuracyBenchmark
import Celeste.ArgumentParse

parser = ArgumentParse.ArgumentParser()
ArgumentParse.add_argument(
    parser,
    "--write-prediction-csv",
    help="Write matched predictions to the given CSV file",
)
ArgumentParse.add_argument(
    parser,
    "ground_truth",
    help="Ground truth CSV catalog",
)
ArgumentParse.add_argument(
    parser,
    "first_predictions",
    help="CSV catalog of predictions",
)
ArgumentParse.add_argument(
    parser,
    "second_predictions",
    help="Second CSV catalog of predictions, to compare to the first",
    required=false,
)
parsed_args = ArgumentParse.parse_args(parser, ARGS)

truth = AccuracyBenchmark.read_catalog(parsed_args["ground_truth"])
prediction_dfs = [AccuracyBenchmark.read_catalog(parsed_args["first_predictions"])]
if haskey(parsed_args, "second_predictions")
    push!(prediction_dfs, AccuracyBenchmark.read_catalog(parsed_args["second_predictions"]))
end
scores = AccuracyBenchmark.score_predictions(truth, prediction_dfs)
println(repr(scores))

if haskey(parsed_args, "write-prediction-csv")
    matched_truth, matched_prediction_dfs = AccuracyBenchmark.match_catalogs(truth, prediction_dfs)
    for prediction_df in matched_prediction_dfs
        for name in names(prediction_df)
            if !in(name, names(truth))
                delete!(prediction_df, name)
            end
        end
    end
    matched_truth[:source] = fill("truth", size(matched_truth, 1))
    for (index, prediction_df) in enumerate(matched_prediction_dfs)
        prediction_df[:source] = fill("prediction $index", size(prediction_df, 1))
    end
    all_predictions = vcat(matched_truth, matched_prediction_dfs...)
    long_df = melt(all_predictions, [:objid, :source])
    long_df[:objid_var] = [
        join([objid, variable], " ")
        for (objid, variable) in zip(long_df[:objid], long_df[:variable])
    ]
    delete!(long_df, :objid)
    delete!(long_df, :variable)
    final_df = unstack(long_df, :objid_var, :source, :value)
    writetable(parsed_args["write-prediction-csv"], final_df)
end
