#!/bin/bash -l

#SBATCH -N 1         #Use 2 nodes
#SBATCH -t 08:00:00  #Set 30 minute time limit
#SBATCH -p regular   #Submit to the regular 'partition'
#SBATCH -L SCRATCH   #Job requires $SCRATCH file system
#SBATCH -C haswell   #Use Haswell nodes
#SBATCH -A dasrepo   # sue dasrepo acct

#srun -n 32 -c 4
# run on a chunk of 64 sources --- execute this 10 times and coalesce
echo "========received range $SOURCE_RANGE"
julia run_celeste_on_field_mcmc.jl --ais-output-dir ais-output-synthetic \
  --initialization-catalog ~/Proj/Celeste.jl/benchmark/accuracy/output/prior_edd9e13e77.csv \
  --use-full-initialization \
  --target-source-range $SOURCE_RANGE \
  --images-jld ~/Proj/Celeste.jl/benchmark/accuracy/output/prior_edd9e13e77_synthetic_7094716d3c.jld
