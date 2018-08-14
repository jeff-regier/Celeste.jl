#!/bin/bash
julia score_mcmc_results.jl \
    --ais-output ais-output-s82 \
    --output-dir s82-results \
    --truth-csv  ~/Proj/Celeste.jl/benchmark/accuracy/output/coadd_for_4263_5_119_d044e0d156.csv \
    --vb-csv     ~/Proj/Celeste.jl/benchmark/accuracy/output/sdss_4263_5_119_predictions_6d8cfdd693.csv \
    --photo-csv  ~/Proj/Celeste.jl/benchmark/accuracy/output/sdss_4263_5_119_primary_b97a8fda22.csv
