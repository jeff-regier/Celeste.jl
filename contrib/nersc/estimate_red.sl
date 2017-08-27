#!/bin/bash -l

#SBATCH --partition=regular
#SBATCH --qos=premium
#SBATCH -N 128 -c 64
#SBATCH --job-name=celeste_estimate
#SBATCH --time=02:00:00
#SBATCH --license=SCRATCH
#SBATCH -C haswell

export CELESTE_STAGE_DIR=$SCRATCH/celeste
export OMP_NUM_THREADS=1
export JULIA_NUM_THREADS=1


module load taskfarmer

export NERSC=$HOME/.julia/v0.6/Celeste/nersc
cd $NERSC

export PATH=$HOME/bin:$PATH
export PATH=$NERSC:$PATH

export THREADS=16

runcommands.sh estimate_tasks_red

