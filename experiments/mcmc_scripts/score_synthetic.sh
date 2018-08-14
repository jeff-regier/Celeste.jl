#!/bin/bash
julia score_mcmc_results.jl \
    --ais-output ais-output-synthetic \
    --output-dir synthetic-results \
    --truth-csv  ~/Proj/Celeste.jl/benchmark/accuracy/output/prior_edd9e13e77.csv \
    --vb-csv     ~/Proj/Celeste.jl/benchmark/accuracy/output/prior_edd9e13e77_synthetic_7094716d3c_predictions_9a93d6bcac.csv
    #--vb-csv     ~/Proj/Celeste.jl/benchmark/accuracy/output/prior_edd9e13e77_synthetic_7094716d3c_predictions_9b69afbd8f.csv
