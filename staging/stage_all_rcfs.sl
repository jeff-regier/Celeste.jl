#!/bin/bash -l


#SBATCH --partition=regular
#SBATCH --qos=premium
#SBATCH --nodes=1
#SBATCH --job-name=celeste
#SBATCH --time=3:00:00
#SBATCH --license=SCRATCH

srun -n 64 sort -R $SCRATCH/all_rcfs | xargs -n 3 -P 64 make
