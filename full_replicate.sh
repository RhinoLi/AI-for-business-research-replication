#!/bin/bash
#SBATCH --job-name=aesthetic_replicate
#SBATCH --partition=gtx4090
#SBATCH --nodes=1               # CRITICAL: Keep GPUs on the same physical machine
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4            # Request 4 GPUs on that single node
#SBATCH --cpus-per-task=12      # Matches your num_data_workers for fast image loading
#SBATCH --mem=128G              # Good for the 112GB dataset overhead
#SBATCH --time=3-00:00:00
#SBATCH --output=production_run_%j.log

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
# sbatch full_replicate.sh
# squeue -u leranli
# tail -f full_replicate.log
# To cancel, can use the following command: scancel
# Check GPU: nvidia-smi
# To kill the program: pkill -u leranli -f python

# python -c "import h5py; import numpy as np; f = h5py.File('/mnt/disk5/home/leranli/project/ms21_product_aesthetic_design_replication_files/data/chair_data_grayscale.h5', 'r'); imgs = f['imgs'][:100]; print(f'Min: {imgs.min()}, Max: {imgs.max()}, Mean: {imgs.mean()}')"