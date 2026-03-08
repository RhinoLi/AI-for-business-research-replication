import h5py
import os

H5_PATH = "/mnt/disk5/home/leranli/project/ms21_product_aesthetic_design_replication_files/data/chair_data_grayscale.h5"

print(f"Inspecting: {H5_PATH}")

try:
    with h5py.File(H5_PATH, 'r') as f:
        print("\n--- Root Keys ---")
        print(list(f.keys()))
        
        # Check the first key found to see its shape
        first_key = list(f.keys())[0]
        print(f"\n--- Shape of '{first_key}' ---")
        print(f[first_key].shape)
        
        # Check dtype
        print(f"Type: {f[first_key].dtype}")

except Exception as e:
    print(f"Error: {e}")