#!/usr/bin/env julia

using DataFrames

import Celeste.AccuracyBenchmark
import Celeste.Infer
import Celeste.ParallelRun

if !(2 <= length(ARGS) <= 3)
    println("Usage: score_predictions.jl <ground truth> <predictions> [predictions]")
    println("Where each argument is a catalog CSV")
end

have_second_predictions = (length(ARGS) == 3)

truth = AccuracyBenchmark.read_catalog(ARGS[1])
first_predictions = AccuracyBenchmark.read_catalog(ARGS[2])
if have_second_predictions
    second_predictions = AccuracyBenchmark.read_catalog(ARGS[3])
else
    second_predictions = first_predictions
end

scores = AccuracyBenchmark.score_predictions(truth, [first_predictions, second_predictions])
if !have_second_predictions
    scores = scores[:, [:N, :first, :field, :source_type]]
end
println(repr(scores))
