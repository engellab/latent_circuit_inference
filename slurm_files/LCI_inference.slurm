#!/bin/bash
#SBATCH --job-name=LCIpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=42G
#SBATCH --time=00:30:00
#SBATCH --mail-user=pt1290@princeton.edu
#SBATCH --output=../log/%x.%A_%a.out
#SBATCH --error=../log/%x.%A_%a.err

# Load modules for CUDA, CUDNN, and conda
module purge
module load cudatoolkit/12.0
module load cudnn/cuda-11.x/8.2.0
module load anaconda3/2022.10

# Activate conda environment
conda activate LCI
echo $1
echo $2
python ../latent_circuit_inference/run_inference/run_LCI.py "+RNN_parent_folder='$1'" "+RNN_subfolder='$2'"