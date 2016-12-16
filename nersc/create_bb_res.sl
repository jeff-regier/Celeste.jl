#!/bin/bash
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -c 2
#SBATCH -t 00:05:00
#SBATCH --license=SCRATCH
#SBATCH -C haswell
#SBATCH --job-name=celestebb

#BB create_persistent name=celestebb capacity=100TB access=striped type=scratch

echo "done create 100 TB"
