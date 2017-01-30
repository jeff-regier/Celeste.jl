#!/bin/bash -l

#SBATCH --partition=debug
#SBATCH --qos=premium
#SBATCH -N 64 -c 64
#SBATCH --job-name=celeste_estimate
#SBATCH --time=00:30:00
#SBATCH --license=SCRATCH
#SBATCH -C haswell

#DW persistentdw name=celestebb

export CELESTE_STAGE_DIR=$DW_PERSISTENT_STRIPED_celestebb/celeste
export OMP_NUM_THREADS=1
export JULIA_NUM_THREADS=1

export JULIA_PKGDIR=$SCRATCH/julia_pkgdir
export MAKEFILE_DIR=$JULIA_PKGDIR/v0.5/Celeste/staging

export PATH=$PATH:/usr/common/tig/taskfarmer/1.5/bin:$(pwd)
export PATH=$SCRATCH/julia/bin:$PATH
export PATH=$JULIA_PKGDIR/v0.5/Celeste/bin:$PATH

export THREADS=16

cd $JULIA_PKGDIR/v0.5/Celeste/staging

runcommands.sh estimate_tasks

