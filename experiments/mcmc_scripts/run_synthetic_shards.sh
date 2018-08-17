#!/bin/bash

declare -a ranges=("1:60" "61:120" "121:180" "181:240" "241:300" "301:360" "361:420" "421:480" "481:500")
#declare -a ranges=("1:60") #"61:120" "121:180" "181:240" "241:300" "301:360" "361:420" "421:480" "481:500")

for r in "${ranges[@]}"
do
    echo "sbatching range $r"
    sbatch --export=SOURCE_RANGE=$r run_synthetic_experiment.sh
done

