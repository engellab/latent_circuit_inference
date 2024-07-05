#!/bin/bash
#SBATCH --job-name=LCI_pt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=42G
#SBATCH --time=2:30:00
#SBATCH --mail-user=pt1290@princeton.edu
#SBATCH --output=../log/%x.%A_%a.out
#SBATCH --error=../log/%x.%A_%a.err

# Load modules for CUDA, CUDNN, and conda
module purge
module load cudatoolkit/12.0
module load cudnn/cuda-11.x/8.2.0
module load anaconda3/2022.10

# Activate conda environment
conda activate RNN_training_pipeline
python ~/latent_circuit_inference/src/set_configs/set_LCI_CDDMtanh_config.py
python ~/latent_circuit_inference/src/run_inference/run_LCI_CDDMtanh.py $1