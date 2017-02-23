#!/usr/bin/env julia

using DataFrames

import Celeste.AccuracyBenchmark
import Celeste.Infer
import Celeste.ParallelRun

if !(2 <= length(ARGS) <= 3)
    println("Usage: score_predictions.jl <ground truth> <predictions> [predictions]")
    println("Where each argument is a catalog CSV")
end

truth = AccuracyBenchmark.read_catalog(ARGS[1])
prediction_dfs = [AccuracyBenchmark.read_catalog(path) for path in ARGS[2:length(ARGS)]]
scores = AccuracyBenchmark.score_predictions(truth, prediction_dfs)
println(repr(scores))
