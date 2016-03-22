#!/bin/bash

source ~/.bashrc

for RANGE in "1:20" "21:40" "41:60" "61:80" "81:100" "101:120" "121:140" "141:160" "161:180" "181:200"; do
  SAFE_RANGE=${RANGE//:/_}
  SH_FILE="/tmp/celeste_cluster_submit_"${SAFE_RANGE}".sh"
  echo "#!/bin/bash" > $SH_FILE
  echo "julia $GIT_REPO_LOC/Celeste.jl/bin/process_image.jl --sources=["$RANGE"]" >> $SH_FILE
  chmod 700 $SH_FILE
done

# TODO: use an argument in the script to make sure this matches.
#qsub $SH_FILE
