#!/bin/bash -l

#SBATCH --partition=regular
#SBATCH --qos=premium
#SBATCH -N 64 -c 64
#SBATCH --job-name=celestebb_stage
#SBATCH --time=03:00:00
#SBATCH --license=SCRATCH
#SBATCH -C haswell

#DW persistentdw name=celestebb

export SDSS_ROOT_DIR="/global/projecta/projectdirs/sdss/data/sdss/dr12/boss"
export FIELD_EXTENTS="/project/projectdirs/dasrepo/celeste-sc16/field_extents.fits"
export CELESTE_STAGE_DIR=$DW_PERSISTENT_STRIPED_celestebb/celeste

export PATH=$PATH:/usr/common/tig/taskfarmer/1.5/bin:$(pwd)

export JULIA_PKGDIR=$SCRATCH/julia_pkgdir
export MAKEFILE_DIR=$JULIA_PKGDIR/v0.5/Celeste/staging
cd $JULIA_PKGDIR/v0.5/Celeste/staging

export THREADS=32

runcommands.sh stage_tasks

