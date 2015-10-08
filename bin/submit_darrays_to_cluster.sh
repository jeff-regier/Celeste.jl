#!/bin/bash

source ~/.bashrc
SH_FILE="/tmp/celeste_cluster_submit.sh"
echo "#!/bin/bash" > $SH_FILE
echo "julia < $GIT_REPO_LOC/Celeste.jl/bin/darray_sandbox.jl" >> $SH_FILE
chmod 700 $SH_FILE

qsub $SH_FILE
