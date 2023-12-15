#!/bin/sh
#SBATCH --account=
#SBATCH --time=72:00:00
#SBATCH --job-name=UnsupervisedSNN_MemoryAnalysis
#SBATCH --partition=accel --gpus=2
#SBATCH --output=MemEstUnsupSTDP.out
#SBATCH --gres=gpu
#SBATCH --mem-per-gpu=64G
module load Python/3.9.5-GCCcore-10.3.0

python ../MemEstUnsupSTDP.py


