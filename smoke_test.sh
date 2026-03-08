#!/bin/bash
#SBATCH --job-name=aesthetic_smoke
#SBATCH --partition=gtx4090
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=smoke_test.log  # This will keep our log name consistent

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
python experiment.py

# cd /mnt/disk5/home/leranli/project/ms21_product_aesthetic_design_replication_files
# sbatch smoke_test.sh
# squeue -u leranli
# tail -f smoke_test.log
# To cancel, can use the following command: scancel