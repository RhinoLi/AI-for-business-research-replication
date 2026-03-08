import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import cv2
import os
from tqdm import tqdm

# ================= CONFIGURATION =================
BASE_PATH = "/mnt/disk5/home/leranli/project/ms21_product_aesthetic_design_replication_files/data"
H5_PATH = os.path.join(BASE_PATH, "chair_data_grayscale.h5") 
INPUT_CSV = os.path.join(os.getcwd(), "om_research_data.csv")
OUTPUT_PLOT = os.path.join(os.getcwd(), "om_efficient_frontier.png")
# =================================================

def calculate_complexity(img_array):
    """
    OPERATIONS AGENT:
    Estimates 'Manufacturing Complexity' using visual entropy and edge density.
    Assumption: More edges/clutter = higher manufacturing cost.
    """
    # 1. Edge Density (Canny) - Represents cuts, joints, and detail
    edges = cv2.Canny(img_array, 100, 200)
    edge_score = np.sum(edges) / 255.0
    
    # 2. Compression Ratio - Represents visual entropy (complex textures don't compress well)
    _, encoded = cv2.imencode('.jpg', img_array, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    compression_score = len(encoded)
    
    # Normalize simply for this index (combining both)
    return edge_score, compression_score

def run_operations_analysis():
    print("\n--- Phase 3: Operations Agent (Complexity Analysis) ---")
    
    if not os.path.exists(INPUT_CSV):
        print("Error: om_research_data.csv not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    complexities = []
    
    print(f"Analyzing manufacturing complexity for {len(df)} designs...")
    
    with h5py.File(H5_PATH, 'r') as f:
        # Determine key
        key = 'IMG_256' if 'IMG_256' in f.keys() else 'IMG_64'
        dataset = f[key]
        
        for index, row in tqdm(df.iterrows(), total=len(df)):
            # Use the correct H5 index we saved earlier
            h5_idx = int(row['h5_index'])
            
            # Extract image (View 30)
            if len(dataset.shape) == 5:
                img = dataset[h5_idx][30]
            else:
                img = dataset[h5_idx]
            
            # Ensure it's 2D for CV2
            if img.shape[-1] == 1:
                img = img.squeeze(-1)
            
            # Calculate Ops Score
            edge, comp = calculate_complexity(img)
            complexities.append(edge) # Using Edge Density as primary proxy

    # Add to DataFrame
    df['ops_complexity_score'] = complexities
    
    # Normalize scores (0-100) for easier plotting
    df['ops_complexity_score'] = (df['ops_complexity_score'] - df['ops_complexity_score'].min()) / \
                                 (df['ops_complexity_score'].max() - df['ops_complexity_score'].min()) * 100
    
    df['market_modernity_score'] = (df['ai_rating_modernity'] - df['ai_rating_modernity'].min()) / \
                                   (df['ai_rating_modernity'].max() - df['ai_rating_modernity'].min()) * 100

    # --- THE TRADEOFF PLOT ---
    plt.figure(figsize=(12, 8))
    
    # Scatter plot
    sns.scatterplot(
        data=df, 
        x='ops_complexity_score', 
        y='market_modernity_score',
        alpha=0.6,
        color='grey'
    )
    
    # Identify "Pareto Efficient" Designs (High Modernity, Low Complexity)
    # Simple heuristic: Top 10% Modernity AND Bottom 50% Complexity
    efficient = df[
        (df['market_modernity_score'] > 80) & 
        (df['ops_complexity_score'] < 40)
    ]
    
    sns.scatterplot(
        data=efficient,
        x='ops_complexity_score',
        y='market_modernity_score',
        color='green',
        s=100,
        label='Efficient Frontier (Target Designs)'
    )
    
    plt.title("The Marketing-Operations Interface: Aesthetic Value vs. Manufacturing Cost", fontsize=14)
    plt.xlabel("Manufacturing Complexity (Estimated Cost)", fontsize=12)
    plt.ylabel("Aesthetic Modernity (Estimated Revenue)", fontsize=12)
    plt.axvline(x=40, color='red', linestyle='--', alpha=0.3, label='Max Complexity Constraint')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(OUTPUT_PLOT)
    print(f"\n✅ Analysis Complete.")
    print(f"📊 Trade-off plot saved to: {OUTPUT_PLOT}")
    print(f"💡 Found {len(efficient)} 'Efficient' designs (High Value / Low Cost).")
    
    # Save updated data
    df.to_csv("om_final_study_results.csv", index=False)

if __name__ == "__main__":
    run_operations_analysis()