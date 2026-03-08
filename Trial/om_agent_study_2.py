import h5py
import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import os

# ================= CONFIGURATION =================
BASE_DATA_PATH = "/mnt/disk5/home/leranli/project/ms21_product_aesthetic_design_replication_files/data"
H5_PATH = os.path.join(BASE_DATA_PATH, "chair_data_grayscale.h5") 
FULL_INDS_PATH = os.path.join(BASE_DATA_PATH, "dining_room_chair_full_inds.csv")
RATINGS_PATH = os.path.join(BASE_DATA_PATH, "chair_ratings_traditional_modern.csv")

OUTPUT_CSV = os.path.join(os.getcwd(), "om_research_data.csv")

# Improved Prompts for Grayscale Data
SCALE_MAPPING = {
    "a photo of a very traditional antique chair": 1.0,
    "a photo of a traditional chair": 2.0,
    "a photo of a standard chair": 3.0,
    "a photo of a modern chair": 4.0,
    "a photo of a futuristic modern chair": 5.0
}
TEXT_PROMPTS = list(SCALE_MAPPING.keys())
SCALE_VALUES = list(SCALE_MAPPING.values())
# =================================================

class AgenticSystem:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing Agents on {self.device}...")
        
        # Load CLIP
        model_id = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        
        # Load Data Map
        self.full_inds = pd.read_csv(FULL_INDS_PATH, header=None, names=['design_id'])
        self.ratings = pd.read_csv(RATINGS_PATH)
        print(f"Loaded Index Map: {len(self.full_inds)} items")

    def get_image_from_h5(self, h5_dataset, index):
        """
        Extracts a representative view (index 30/62) from the H5 dataset.
        """
        try:
            # FIX: Check if index is within bounds
            if index >= h5_dataset.shape[0]:
                print(f"Warning: Index {index} out of bounds for H5 shape {h5_dataset.shape}")
                return None

            # Handle Multi-View Shape (N, 62, 128, 128, 1)
            raw_img = h5_dataset[index]
            
            if len(raw_img.shape) == 4: # (62, 128, 128, 1)
                img_array = raw_img[30] # Pick view 30
            else:
                img_array = raw_img # Fallback
            
            # Squeeze and Convert
            img_array = img_array.squeeze(-1)
            return Image.fromarray(img_array, 'L').convert('RGB')
            
        except Exception as e:
            print(f"Error extracting image {index}: {e}")
            return None

    def market_agent_rate(self):
        print("\n--- Phase 1: Market Agent Assessment ---")
        results = []
        VIEWS = [0, 15, 30, 45] # Sample 4 angles

        try:
            with h5py.File(H5_PATH, 'r') as f:
                # Use IMG_256 for best quality
                dataset_name = 'IMG_256' if 'IMG_256' in f.keys() else 'IMG_64'
                print(f"Using Image Source: {dataset_name} | Shape: {f[dataset_name].shape}")
                dataset = f[dataset_name]

                print("Agent is scoring inventory...")
                for i in tqdm(range(len(self.full_inds))):
                    # --- CRITICAL FIX: Use design_id as H5 index ---
                    design_id = int(self.full_inds.iloc[i]['design_id'])
                    
                    # 1. Get Human Rating
                    human_rating = np.nan
                    row = self.ratings[self.ratings['design_id'] == design_id]
                    if not row.empty:
                        human_rating = row.iloc[0]['real_value']

                    # 2. Visual Analysis (Multi-View)
                    view_probs = []
                    valid_image = True
                    
                    for v in VIEWS:
                        try:
                            # Use design_id directly to grab the correct chair
                            img_arr = dataset[design_id][v].squeeze(-1)
                            img = Image.fromarray(img_arr, 'L').convert('RGB')
                            
                            inputs = self.processor(text=TEXT_PROMPTS, images=img, return_tensors="pt", padding=True).to(self.device)
                            with torch.no_grad():
                                outputs = self.model(**inputs)
                                probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
                                view_probs.append(probs)
                        except Exception as e:
                            valid_image = False
                            break
                    
                    if not valid_image or not view_probs:
                        continue

                    # Average probabilities
                    avg_probs = np.mean(view_probs, axis=0)
                    ai_rating = np.sum(avg_probs * SCALE_VALUES)

                    # 3. Save with CORRECT Mapping
                    results.append({
                        'h5_index': design_id,  # SAVE TRUE ID
                        'design_id': design_id,
                        'human_rating_modernity': human_rating,
                        'ai_rating_modernity': ai_rating
                    })

        except Exception as e:
            print(f"Error reading H5: {e}")
            return None

        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"✅ Ratings saved to: {OUTPUT_CSV}")
        return df

    def designer_agent_generate(self, target_rating=4.0, mode='retrieval'):
        print(f"\n--- Phase 2: Design Agent ({mode}) ---")
        
        if mode == 'retrieval':
            if not os.path.exists(OUTPUT_CSV):
                print("Error: Run market_agent_rate() first.")
                return

            df = pd.read_csv(OUTPUT_CSV)
            # Find closest match
            df['diff'] = (df['ai_rating_modernity'] - target_rating).abs()
            best_match = df.iloc[df['diff'].idxmin()]
            
            design_id = int(best_match['design_id'])
            actual_score = best_match['ai_rating_modernity']
            
            print(f"Selected Design ID: {design_id} (AI Score: {actual_score:.2f})")
            
            with h5py.File(H5_PATH, 'r') as f:
                dataset_name = 'IMG_256' if 'IMG_256' in f.keys() else 'IMG_64'
                # Use TRUE ID
                img = self.get_image_from_h5(f[dataset_name], int(best_match['h5_index']))
                
                if img:
                    filename = f"design_ID{design_id}_score{actual_score:.1f}.png"
                    save_path = os.path.join(os.getcwd(), filename)
                    img.save(save_path)
                    print(f"✅ Retrieved Image Saved: {save_path}")

        elif mode == 'generation':
            prompt = "a futuristic ultra-modern chair, minimal, white background, 8k" if target_rating > 3 else "a antique wooden windsor chair, ornate, 8k"
            print(f"Generating: '{prompt}'")
            try:
                from diffusers import StableDiffusionPipeline
                pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
                pipe = pipe.to(self.device)
                image = pipe(prompt).images[0]
                save_path = os.path.join(os.getcwd(), "gen_design.png")
                image.save(save_path)
                print(f"✅ Generated Image Saved: {save_path}")
            except ImportError:
                print("Diffusers not installed.")

if __name__ == "__main__":
    system = AgenticSystem()
    df = system.market_agent_rate()
    if df is not None:
        system.designer_agent_generate(target_rating=4.5, mode='retrieval')