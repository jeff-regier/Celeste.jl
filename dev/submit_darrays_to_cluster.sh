#!/bin/bash

source ~/.bashrc
SH_FILE="/tmp/celeste_cluster_submit.sh"
echo "#!/bin/bash" > $SH_FILE
echo "julia < $GIT_REPO_LOC/Celeste.jl/bin/darray_sandbox.jl" >> $SH_FILE
chmod 700 $SH_FILE

# TODO: use an argument in the script to make sure this matches.
export OMP_NUM_THREADS=10
qsub $SH_FILE
