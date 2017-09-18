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

# the code below is run only if the user specifies the --write-prediction-csv
# flag. If present, we write out a csv file that is useful for debugging
# individual predictions. In that file, coadd, primary, and celeste are matched
# up for every light source.
if haskey(parsed_args, "write-prediction-csv")
    matched_truth, matched_prediction_dfs = AccuracyBenchmark.match_catalogs(truth, prediction_dfs)
    for prediction_df in matched_prediction_dfs
        for name in names(prediction_df)
            if !in(name, names(truth))
                delete!(prediction_df, name)
            end
        end
    end
    matched_truth[:index] = collect(1:nrow(matched_truth))
    matched_truth[:source] = fill("truth", size(matched_truth, 1))
    for (i, prediction_df) in enumerate(matched_prediction_dfs)
        prediction_df[:index] = collect(1:nrow(prediction_df))
        prediction_df[:source] = fill("prediction $i", size(prediction_df, 1))
    end
    all_predictions = vcat(matched_truth, matched_prediction_dfs...)
    long_df = melt(all_predictions, [:index, :source])
    long_df[:index_var] = [
        join([idx, variable], " ")
        for (idx, variable) in zip(long_df[:index], long_df[:variable])
    ]
    delete!(long_df, :index)
    delete!(long_df, :variable)

    # The `index_var` column has the format "source_id property_name". In
    # `long_df`, typically all three sets of predictions have one prediction
    # for any particular source_id and property name. Below we unstack to
    # put all three predictions for a particular (source_id, property) in the
    # same row.
    final_df = unstack(long_df, :index_var, :source, :value)

    # Next we split index_var into two columns: source_id and property.
    # It's easier to work with subsequently this way.
    final_df[:, :source_id] = [parse(Int, match(r"^\S+", x).match) for x in final_df[:, :index_var]]
    final_df[:, :property] = [match(r"[a-z].*", x).match for x in final_df[:, :index_var]]
    delete!(final_df, :index_var)

    writetable(parsed_args["write-prediction-csv"], final_df)
end
