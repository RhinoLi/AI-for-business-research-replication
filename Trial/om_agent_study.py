import h5py
import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import os

# ================= CONFIGURATION =================
# 1. Data Paths (Update these if filenames differ)
base_data_path = "/mnt/disk5/home/leranli/project/ms21_product_aesthetic_design_replication_files/data"
     

H5_PATH = os.path.join(base_data_path, "chair_data_grayscale.h5") 
FULL_INDS_PATH = os.path.join(base_data_path, "dining_room_chair_full_inds.csv")
RATINGS_PATH = os.path.join(base_data_path, "chair_ratings_traditional_modern.csv")

# 2. Output
OUTPUT_CSV = os.path.join(os.getcwd(), "om_research_data.csv")

# 3. Market Agent Prompts (The "Survey Questions")
TEXT_PROMPTS = [
    "a modern chair",
    "a traditional chair",
    "a minimalist chair", 
    "a complex chair",
    "a luxurious chair"
]
# =================================================

class AgenticSystem:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing Agents on {self.device}...")
        
        # Load CLIP (The Brain)
        model_id = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        
        # Load Data Map
        self.full_inds = pd.read_csv(FULL_INDS_PATH, header=None, names=['design_id'])
        self.ratings = pd.read_csv(RATINGS_PATH)
        print(f"Loaded Index Map: {len(self.full_inds)} items")
        print(f"Loaded Ratings: {len(self.ratings)} items")

    def get_image_from_h5(self, index, h5_dataset):
        """
        Extracts and converts a single representative view from the H5 dataset.
        Handles the shape (62, H, W, 1).
        """
        # Shape of one item is (62, H, W, 1) containing 62 views
        # We pick view 30 (roughly 180 degrees/front-ish) for the "Hero Shot"
        view_idx = 30 
        
        # Extract specific view
        img_array = h5_dataset[index][view_idx] # Shape: (H, W, 1)
        
        # Remove the single channel dimension -> (H, W)
        img_array = img_array.squeeze(-1) 
        
        # Convert Grayscale (L) to RGB so it saves correctly as a PNG
        return Image.fromarray(img_array, 'L').convert('RGB')

    def market_agent_rate(self):
        """
        PHASE 1: THE MARKET AGENT (FIXED FOR MULTI-VIEW H5)
        Scores products by averaging visual cues from multiple angles.
        """
        print("\n--- Phase 1: Market Agent Assessment (Multi-View) ---")
        results = []
        
        # We will sample 4 views to get a 360-degree assessment
        # Indices 0, 15, 30, 45 roughly correspond to 0, 90, 180, 270 degrees in a 62-frame rotation
        VIEWS_TO_CHECK = [0, 15, 30, 45] 
        
        try:
            with h5py.File(H5_PATH, 'r') as f:
                # FIX 1: Use the correct key. 'IMG_256' is best for features.
                dataset_name = 'IMG_256' if 'IMG_256' in f.keys() else 'IMG_64'
                print(f"Using resolution: {dataset_name}")
                images_dataset = f[dataset_name]
                
                print(f"Agent is scoring inventory ({len(self.full_inds)} items)...")
                
                # Iterate ONLY through the 700 valid chairs in your CSV
                for _, row in tqdm(self.full_inds.iterrows(), total=len(self.full_inds)):
                    
                    # FIX 2: Map CSV design_id to H5 index
                    # Assuming design_id corresponds to the index in the H5 file
                    h5_idx = int(row['design_id'])
                    design_id = h5_idx
                    
                    # Get Ground Truth Rating
                    human_rating = np.nan
                    rating_row = self.ratings[self.ratings['design_id'] == design_id]
                    if not rating_row.empty:
                        human_rating = rating_row.iloc[0]['real_value']
                    
                    # FIX 3: Process Multi-View Data
                    # Shape is (62, H, W, 1)
                    raw_object_data = images_dataset[h5_idx] 
                    
                    # Accumulator for scores across views
                    view_scores = []
                    
                    for view_idx in VIEWS_TO_CHECK:
                        # Extract specific view (H, W, 1)
                        img_array = raw_object_data[view_idx]
                        
                        # Remove last dim -> (H, W)
                        img_array = img_array.squeeze(-1) 
                        
                        # Convert Grayscale (L) to RGB for CLIP
                        image = Image.fromarray(img_array, 'L').convert('RGB')
                        
                        inputs = self.processor(
                            text=TEXT_PROMPTS, 
                            images=image, 
                            return_tensors="pt", 
                            padding=True
                        ).to(self.device)

                        with torch.no_grad():
                            outputs = self.model(**inputs)
                            # Get probabilities
                            probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
                            view_scores.append(probs)
                    
                    # Average the scores across the 4 views for robustness
                    avg_scores = np.mean(view_scores, axis=0)

                    # 4. Record Data
                    data_row = {
                        'h5_index': h5_idx,
                        'design_id': design_id,
                        'human_rating_modernity': human_rating,
                    }
                    
                    for prompt_idx, prompt in enumerate(TEXT_PROMPTS):
                        clean_col = "ai_score_" + prompt.replace("a ", "").replace(" ", "_")
                        data_row[clean_col] = avg_scores[prompt_idx]
                        
                    results.append(data_row)
                    
        except KeyError as e:
            print(f"H5 Key Error: {e}. Check H5 keys with inspect script.")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Assessment Complete. Data saved to {OUTPUT_CSV}")
        return df

def designer_agent_generate(self, prompt, mode='retrieval'):
        """
        PHASE 2: THE DESIGNER AGENT
        Either retrieves a matching design from inventory OR generates a new one.
        """
        print(f"\n--- Phase 2: Design Agent ({mode}) ---")
        print(f"Request: '{prompt}'")
        
        if mode == 'retrieval':
            # Use the data we just generated to find the best match
            if not os.path.exists(OUTPUT_CSV):
                print("Run market_agent_rate() first to index the data.")
                return
            
            df = pd.read_csv(OUTPUT_CSV)
            
            # Map prompt to the column name
            target_col = "ai_score_" + prompt.replace("a ", "").replace(" ", "_")
            
            if target_col in df.columns:
                best_match = df.iloc[df[target_col].idxmax()]
                design_id = int(best_match['design_id'])
                score = best_match[target_col]
                
                print(f"Selected Design ID: {design_id}")
                print(f"Confidence Score: {score:.4f}")
                
                # Retrieve and Save Image
                with h5py.File(H5_PATH, 'r') as f:
                    # FIX: Dynamically find the correct key ('IMG_256' or 'IMG_64')
                    # just like we did in Phase 1
                    dataset_name = 'IMG_256' if 'IMG_256' in f.keys() else 'IMG_64'
                    
                    # Pass the specific dataset object (e.g., f['IMG_256'])
                    img = self.get_image_from_h5(int(best_match['h5_index']), f[dataset_name])
                    
                    save_path = f"selected_design_{design_id}.png"
                    img.save(save_path)
                    print(f"Design saved to: {save_path}")
            else:
                print(f"Construct '{prompt}' not found in pre-computed scores.")

        elif mode == 'generation':
            # This requires 'diffusers' library installed
            print("Initializing Stable Diffusion...")
            try:
                from diffusers import StableDiffusionPipeline
                pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
                pipe = pipe.to(self.device)
                
                image = pipe(prompt).images[0]
                image.save("generated_design.png")
                print("New design generated and saved to generated_design.png")
            except ImportError:
                print("Error: 'diffusers' library not found. Please run: pip install diffusers accelerators")

# ================= EXECUTION =================
if __name__ == "__main__":
    system = AgenticSystem()
    
    # 1. Run Market Analysis (Rate)
    # This creates the dataset for your regression paper
    df = system.market_agent_rate()
    
    if df is not None:
        # 2. Run Design Selection (Generate/Retrieve)
        # Example: "Marketing wants a modern chair"
        system.designer_agent_generate("a modern chair", mode='retrieval')
        
        # Example: "Marketing wants a modern chair"
        system.designer_agent_generate("a very fancy  chair", mode='generation')

        # Example: "Operations wants a simple chair"
        # (Assuming 'minimalist' is a proxy for simple/manufacturable)
        system.designer_agent_generate("a minimalist chair", mode='retrieval')