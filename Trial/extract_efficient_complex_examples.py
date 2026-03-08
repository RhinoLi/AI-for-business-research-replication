import pandas as pd
import numpy as np
import h5py
import cv2
import os
from PIL import Image
from tqdm import tqdm

# ================= CONFIGURATION =================
# Paths (Adjust to match your setup)
BASE_PATH = "/mnt/disk5/home/leranli/project/ms21_product_aesthetic_design_replication_files/data"
H5_PATH = os.path.join(BASE_PATH, "chair_data_grayscale.h5") 

# Input: Tries to find the final results first, falls back to raw data
FINAL_CSV = "om_final_study_results.csv"
RAW_CSV = "om_research_data.csv"

# Output
OUTPUT_DIR = "paper_examples"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# =================================================

def calculate_complexity(img_array):
    """Recalculate complexity if missing from CSV"""
    edges = cv2.Canny(img_array, 100, 200)
    edge_score = np.sum(edges) / 255.0
    return edge_score

def get_image_from_h5(h5_file, index):
    """Robust image extraction"""
    # Try typical keys
    keys = ['IMG_256', 'IMG_128', 'IMG_64', 'data']
    dataset = None
    for k in keys:
        if k in h5_file:
            dataset = h5_file[k]
            break
            
    if dataset is None: return None

    # Handle (N, Views, H, W, C) or (N, H, W, C)
    img_data = dataset[index]
    if len(img_data.shape) == 4: 
        img_data = img_data[30] # View 30 (Front/Side)
    
    if img_data.shape[-1] == 1:
        img_data = img_data.squeeze(-1)
        return Image.fromarray(img_data, 'L').convert('RGB')
    return Image.fromarray(img_data, 'RGB')

def main():
    print("--- Extracting Exemplars for Paper ---")
    
    # 1. Load Data
    if os.path.exists(FINAL_CSV):
        print(f"Loading analyzed data from {FINAL_CSV}...")
        df = pd.read_csv(FINAL_CSV)
    elif os.path.exists(RAW_CSV):
        print(f"Loading raw data from {RAW_CSV} and calculating complexity...")
        df = pd.read_csv(RAW_CSV)
        
        # Calculate scores on the fly
        complexities = []
        with h5py.File(H5_PATH, 'r') as f:
            # Find key
            key = 'IMG_256' if 'IMG_256' in f.keys() else 'IMG_64'
            dataset = f[key]
            
            for index, row in tqdm(df.iterrows(), total=len(df)):
                h5_idx = int(row['h5_index'])
                
                # Extract image for calculation
                if len(dataset.shape) == 5:
                    img = dataset[h5_idx][30].squeeze(-1)
                else:
                    img = dataset[h5_idx].squeeze(-1)
                    
                score = calculate_complexity(img)
                complexities.append(score)
        
        df['ops_complexity_score'] = complexities
        
        # Normalize
        df['ops_complexity_score'] = (df['ops_complexity_score'] - df['ops_complexity_score'].min()) / \
                                     (df['ops_complexity_score'].max() - df['ops_complexity_score'].min()) * 100
        df['market_modernity_score'] = (df['ai_rating_modernity'] - df['ai_rating_modernity'].min()) / \
                                       (df['ai_rating_modernity'].max() - df['ai_rating_modernity'].min()) * 100
    else:
        print("Error: No CSV found. Please run the previous analysis script first.")
        return

    # 2. Identify Groups
    
    # Group A: The "Efficient Frontier" (High Modernity, Low Complexity)
    # Filter: Top 20% Modernity AND Bottom 20% Complexity
    efficient_candidates = df[
        (df['market_modernity_score'] > 80) & 
        (df['ops_complexity_score'] < 20)
    ].sort_values(by='market_modernity_score', ascending=False)
    
    # Group B: Maximum Complexity (High Cost)
    complex_candidates = df.sort_values(by='ops_complexity_score', ascending=False)

    # Select Top 5 from each
    top_efficient = efficient_candidates.head(5)
    top_complex = complex_candidates.head(5)

    print(f"\nFound {len(efficient_candidates)} efficient designs.")
    print(f"Selecting Top 5 Efficient and Top 5 Complex examples...")

    # 3. Save Images
    with h5py.File(H5_PATH, 'r') as f:
        # Save Efficient
        print("\n--- Saving Efficient Frontier Examples ---")
        for i, (_, row) in enumerate(top_efficient.iterrows()):
            design_id = int(row['design_id'])
            mod_score = row['market_modernity_score']
            ops_score = row['ops_complexity_score']
            
            img = get_image_from_h5(f, int(row['h5_index']))
            if img:
                fname = f"Efficient_{i+1}_ID{design_id}_Mod{mod_score:.0f}_Ops{ops_score:.0f}.png"
                img.save(os.path.join(OUTPUT_DIR, fname))
                print(f"Saved: {fname} (Modernity: {mod_score:.1f}, Complexity: {ops_score:.1f})")

        # Save Complex
        print("\n--- Saving Max Complexity Examples ---")
        for i, (_, row) in enumerate(top_complex.iterrows()):
            design_id = int(row['design_id'])
            mod_score = row['market_modernity_score']
            ops_score = row['ops_complexity_score']
            
            img = get_image_from_h5(f, int(row['h5_index']))
            if img:
                fname = f"Complex_{i+1}_ID{design_id}_Mod{mod_score:.0f}_Ops{ops_score:.0f}.png"
                img.save(os.path.join(OUTPUT_DIR, fname))
                print(f"Saved: {fname} (Modernity: {mod_score:.1f}, Complexity: {ops_score:.1f})")

    print(f"\n✅ Done! Check the '{OUTPUT_DIR}' folder for your images.")

if __name__ == "__main__":
    main()