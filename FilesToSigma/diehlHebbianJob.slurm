#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-iel
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --job-name="bindsNet custom network SNN analysis of memory and accuracy 3"
#SBATCH --output=diehlAndCookHebbianCPU_nograd.out
#SBATCH --mem=150G
module load Python/3.8.6-GCCcore-10.2.0

python diehlAndCookHebbian.py

