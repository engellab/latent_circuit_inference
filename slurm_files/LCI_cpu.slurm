#!/bin/bash
#SBATCH --job-name=LCI_pt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=42G
#SBATCH --time=01:00:00
#SBATCH --mail-user=pt1290@princeton.edu
#SBATCH --output=../log/%x.%A_%a.out
#SBATCH --error=../log/%x.%A_%a.err

# Load modules
module purge
module load anaconda3/2022.5

# Activate conda environment
conda activate RNN_training_pipeline
python ~/latent_circuit_inference/src/set_LCI_config_8nodes.py
python ~/latent_circuit_inference/src/run_LCI.py $1