#!/bin/bash -l


#SBATCH --partition=debug
#SBATCH --nodes=2
#SBATCH --job-name=celeste
#SBATCH --time=00:30:00
#SBATCH --license=SCRATCH


export OMP_NUM_THREADS=1
export JULIA_NUM_THREADS=32
export SDSS_ROOT_DIR="/global/projecta/projectdirs/sdss/data/sdss/dr12/boss"
export FIELD_EXTENTS="/project/projectdirs/dasrepo/celeste-sc16/field_extents.fits"
module load impi
export I_MPI_PIN_DOMAIN=auto
export USE_DTREE=1
export I_MPI_PMI_LIBRARY=/usr/lib64/slurmpmi/libpmi.so
srun -n 32 $SCRATCH/julia/bin/julia-debug --depwarn=no "$HOME/.julia/v0.5/Celeste/bin/celeste.jl" infer-box 200 200.5 38.1 38.35 $SCRATCH/multi
