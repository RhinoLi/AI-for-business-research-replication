#!/bin/bash
#SBATCH --job-name=aesthetic_smoke
#SBATCH --partition=gtx4090
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=extension_%j.log  # This will keep our log name consistent

# 1. Clear any inherited environment variables
conda deactivate
conda deactivate

# 2. Source the central Conda initialization script
source /mnt/disk5/software/miniconda3/etc/profile.d/conda.sh

# 3. Activate the environment by its FULL PATH to be 100% safe
conda activate /mnt/disk5/home/leranli/.conda/envs/aesthetics

# 4. Verify we are in the right place
which python

# 5. Run the experiment
python om_agent_study_2.py
python evaluate_and_generate.py
python om_tradeoff_analysis.py
python extract_efficient_complex_examples.py
# cd /mnt/disk5/home/leranli/project/ms21_product_aesthetic_design_replication_files/Trial
# sbatch smoke_trail.sh
# squeue -u leranli

# To cancel, can use the following command: scancel