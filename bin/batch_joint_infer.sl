#!/bin/bash -l


#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --job-name=celeste
#SBATCH --time=02:00:00
#SBATCH --license=SCRATCH


export SDSS_ROOT_DIR="/global/projecta/projectdirs/sdss/data/sdss/dr12/boss"
export FIELD_EXTENTS="/project/projectdirs/dasrepo/celeste-sc16/field_extents.fits"
export CELESTE_STAGE_DIR=/global/cscratch1/sd/jregier/celeste
export USE_DTREE=0

module load impi
export I_MPI_PIN_DOMAIN=auto
export I_MPI_PMI_LIBRARY=/usr/lib64/slurmpmi/libpmi.so


#export JULIA_NUM_THREADS=1
#srun -n 1 $HOME/julia-3c9d75391c/bin/julia ./test/runtests.jl joint_infer >& infer_multi_iter_1_thread
export JULIA_NUM_THREADS=8
srun -n 1 $HOME/julia-3c9d75391c/bin/julia ./test/runtests.jl joint_infer >& infer_multi_iter_8_thread
export JULIA_NUM_THREADS=12
srun -n 1 $HOME/julia-3c9d75391c/bin/julia ./test/runtests.jl joint_infer >& infer_multi_iter_12_thread

