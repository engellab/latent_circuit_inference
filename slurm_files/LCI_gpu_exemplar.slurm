#!/bin/bash
#SBATCH --job-name=LCI_pt_$1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=42G
#SBATCH --time=3:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=pt1290@princeton.edu
#SBATCH --output=../log/%x.%A_%a.out
#SBATCH --error=../log/%x.%A_%a.err

# Load modules for CUDA, CUDNN, and conda
module purge
module load cudatoolkit/12.0
module load cudnn/cuda-11.x/8.2.0
module load anaconda3/2022.5

# Activate conda environment
conda activate RNN_training_pipeline

python ~/latent_circuit_inference/src/set_LCI_config.py
python ~/latent_circuit_inference/src/fittingLC_to_exemplar_RNN.py $1