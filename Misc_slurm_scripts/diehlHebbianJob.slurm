#!/bin/sh
#SBATCH --account=
#SBATCH --time=24:00:00
#SBATCH --job-name=UnsupervisedSNNHebbian
#SBATCH --partition=accel --gpus=2
#SBATCH --output=diehlAndCookHebbian.out
#SBATCH --gres=gpu
#SBATCH --mem-per-gpu=64G
module load Python/3.11.3-GCCcore-12.3.0

python ../diehlAndCookHebbian.py


