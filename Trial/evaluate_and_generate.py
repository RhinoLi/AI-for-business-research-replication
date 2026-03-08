import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from PIL import Image
import os
from scipy.stats import pearsonr, spearmanr

# ================= CONFIGURATION =================
# Paths (Adjust if necessary)
BASE_PATH = os.getcwd()
RESULTS_CSV = os.path.join(BASE_PATH, "om_research_data.csv")
H5_PATH = os.path.join("/mnt/disk5/home/leranli/project/ms21_product_aesthetic_design_replication_files/data/chair_data_grayscale.h5")

# Output directory for images
IMG_OUT_DIR = os.path.join(BASE_PATH, "agent_samples")
os.makedirs(IMG_OUT_DIR, exist_ok=True)
# =================================================

def analyze_and_plot():
    """
    Part 1: Compare AI Ratings with Ground Truth
    Generates statistical metrics and a regression plot.
    """
    print("\n--- Part 1: Analytical Verification ---")
    
    if not os.path.exists(RESULTS_CSV):
        print(f"Error: {RESULTS_CSV} not found. Run the previous script first.")
        return

    df = pd.read_csv(RESULTS_CSV)
    
    # Filter for rows where we have human ratings
    df_clean = df.dropna(subset=['human_rating_modernity'])
    
    if len(df_clean) == 0:
        print("Warning: No matched human ratings found in the CSV.")
        print("Here are the top 5 modern chairs identified by AI instead:")
        print(df.sort_values(by='ai_rating_modernity', ascending=False)[['design_id', 'ai_rating_modernity']].head())
        return

    # 1. Statistics
    human = df_clean['human_rating_modernity']
    ai = df_clean['ai_rating_modernity']
    
    p_corr, p_val = pearsonr(human, ai)
    s_corr, s_val = spearmanr(human, ai)
    
    print(f"Sample Size: {len(df_clean)} chairs")
    print(f"Pearson Correlation (Linear):  r={p_corr:.4f} (p={p_val:.4e})")
    print(f"Spearman Correlation (Rank):   rho={s_corr:.4f} (p={s_val:.4e})")
    
    if p_val < 0.05:
        print("✅ RESULT: Significant positive correlation detected. The AI Agent aligns with human perception.")
    else:
        print("⚠️ RESULT: Correlation not significant. Check data alignment.")

    # 2. Visualization
    plt.figure(figsize=(10, 6))
    sns.regplot(x=human, y=ai, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
    plt.title(f'Agent Alignment: Human vs AI Modernity Ratings\n(r={p_corr:.2f}, N={len(df_clean)})')
    plt.xlabel('Human Ground Truth (1-5 Scale)')
    plt.ylabel('AI Agent Rating (1-5 Scale)')
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(BASE_PATH, "alignment_regression_plot.png")
    plt.savefig(plot_path)
    print(f"📊 Regression plot saved to: {plot_path}")

def get_image_from_h5(h5_file, index):
    """Helper to extract image from your specific H5 structure"""
    # Try different keys typical for ProGAN
    keys_to_try = ['IMG_256', 'IMG_128', 'IMG_64', 'data']
    dataset = None
    for k in keys_to_try:
        if k in h5_file:
            dataset = h5_file[k]
            break
            
    if dataset is None:
        raise KeyError(f"Could not find image data. Keys found: {list(h5_file.keys())}")

    # Handle (N, Views, H, W, C) or (N, H, W, C)
    img_data = dataset[index]
    
    if len(img_data.shape) == 4: # (Views, H, W, C)
        # Pick view 30 (approx 180 degrees/front)
        img_data = img_data[30]
    
    # Squeeze channel dim if grayscale (H, W, 1) -> (H, W)
    if img_data.shape[-1] == 1:
        img_data = img_data.squeeze(-1)
        return Image.fromarray(img_data, 'L').convert('RGB')
    else:
        return Image.fromarray(img_data, 'RGB')

def generate_samples():
    """
    Part 2: Generate/Retrieve Sample Chairs
    Retrieves the actual images for the top/bottom ranked chairs.
    """
    print("\n--- Part 2: Generating Sample Images ---")
    
    df = pd.read_csv(RESULTS_CSV)
    
    # Sort by AI rating
    df_sorted = df.sort_values(by='ai_rating_modernity', ascending=False)
    
    # Pick Top 3 (Modern) and Bottom 3 (Traditional)
    top_3 = df_sorted.head(3)
    bottom_3 = df_sorted.tail(3)
    
    samples_to_extract = pd.concat([top_3, bottom_3])
    
    try:
        with h5py.File(H5_PATH, 'r') as f:
            print(f"Extracting images from {H5_PATH}...")
            
            for _, row in samples_to_extract.iterrows():
                design_id = int(row['design_id'])
                h5_idx = int(row['h5_index'])
                rating = row['ai_rating_modernity']
                
                # Determine Label
                label = "Modern" if rating > 3.0 else "Traditional"
                
                # Get Image
                img = get_image_from_h5(f, h5_idx)
                
                # Save
                filename = f"{label}_ID{design_id}_Rating{rating:.2f}.png"
                save_path = os.path.join(IMG_OUT_DIR, filename)
                img.save(save_path)
                print(f"Saved: {save_path}")
                
    except Exception as e:
        print(f"Error extracting H5 images: {e}")

    # Stable Diffusion Generation (Optional)
    print("\n--- Part 3: Agentic Generation (Stable Diffusion) ---")
    try:
        from diffusers import StableDiffusionPipeline
        import torch
        
        print("Stable Diffusion Library found. Generating synthetic concepts...")
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")

        prompts = {
            "Agent_Generated_Modern": "a ultra-modern futuristic chair, gray scale, 8k",
            "Agent_Generated_Traditional": "a antique wooden windsor chair, gray scale, 8k"
        }

        for name, prompt in prompts.items():
            print(f"Generating: '{prompt}'...")
            image = pipe(prompt).images[0]
            save_path = os.path.join(IMG_OUT_DIR, f"{name}.png")
            image.save(save_path)
            print(f"Saved: {save_path}")

    except ImportError:
        print("Note: 'diffusers' library not installed. Skipping synthetic generation.")
        print("To enable this: pip install diffusers transformers accelerate")

if __name__ == "__main__":
    analyze_and_plot()
    generate_samples()