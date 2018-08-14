#!/bin/bash

declare -a ranges=("1:60" "61:120" "121:180" "181:240" "241:300" "301:360" "361:420" "421:480" "481:540" "541:600" "601:660" "661:720" "721:780" "781:841")

for r in "${ranges[@]}"
do
    echo "sbatching range $r"
    sbatch --export=SOURCE_RANGE=$r run_s82_experiment.sh
done

