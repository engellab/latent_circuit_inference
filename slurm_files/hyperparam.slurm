#!/bin/bash
#SBATCH --job-name=LCI_hyperparam_ptolmachev
#SBATCH --nodes=10
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=pt1290@princeton.edu
#SBATCH --output=../log/%x.%A_%a.out
#SBATCH --error=../log/%x.%A_%a.err

# Load modules for CUDA, CUDNN, and conda
module purge
module load anaconda3/2022.5

# Activate conda environment
conda activate RNN_training_pipeline

python ~/latent_circuit_inference/src/set_LCI_config.py
python ~/latent_circuit_inference/src/hyperparam_sweep.py