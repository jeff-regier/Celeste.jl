#!/bin/bash -l

#SBATCH --partition=regular
#SBATCH --qos=premium
#SBATCH -N 64 -c 64
#SBATCH --job-name=celestebb_stage
#SBATCH --time=05:00:00
#SBATCH --license=SCRATCH
#SBATCH -C haswell

#DW persistentdw name=celestebb

export SDSS_ROOT_DIR="/global/projecta/projectdirs/sdss/data/sdss/dr12/boss"
export FIELD_EXTENTS="/project/projectdirs/dasrepo/celeste-sc16/field_extents.fits"
export CELESTE_STAGE_DIR=$DW_PERSISTENT_STRIPED_celestebb/celeste

module load taskfarmer

export NERSC=$HOME/.julia/v0.6/Celeste/nersc
cd $NERSC

export THREADS=32

runcommands.sh stage_tasks

