#!/usr/bin/env julia

using DataFrames
import JLD

import Celeste.AccuracyBenchmark
import Celeste.Log
import Celeste.ParallelRun

data_rows = DataFrame[]
for jld_path in ARGS
    Log.info("Reading $jld_path...")
    sources = JLD.load(jld_path)["results"]
    Log.info("Found $(length(sources)) sources")
    for source in sources
        push!(data_rows,
              AccuracyBenchmark.variational_parameters_to_data_frame_row(
                  source.vs)
              )
    end
end
data = vcat(data_rows...)

if length(ARGS) == 1
    csv_path = string(splitext(ARGS[1])[1], ".csv")
else
    csv_path = "celeste_catalog.csv"
end
Log.info("Writing $csv_path...")
writetable(csv_path, data)
