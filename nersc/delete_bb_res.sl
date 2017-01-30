#!/bin/bash
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -c 2
#SBATCH -t 00:05:00
#SBATCH -C haswell

#BB destroy_persistent name=celestebb

echo "done destroy"
