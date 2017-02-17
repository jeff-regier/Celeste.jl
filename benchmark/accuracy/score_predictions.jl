#!/usr/bin/env julia

using DataFrames

import Celeste.AccuracyBenchmark
import Celeste.Infer
import Celeste.ParallelRun

if length(ARGS) != 3
    println("Usage: score_predictions.jl <ground truth> <prediction> <prediction>")
    println("Where each argument is a catalog CSV")
end

truth = AccuracyBenchmark.read_catalog(ARGS[1])
first_predictions = AccuracyBenchmark.read_catalog(ARGS[2])
second_predictions = AccuracyBenchmark.read_catalog(ARGS[3])

scores = AccuracyBenchmark.score_predictions(truth, [first_predictions, second_predictions])
println(repr(scores))
