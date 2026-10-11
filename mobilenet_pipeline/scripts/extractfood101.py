import os
import shutil
from datasets import load_dataset
from tqdm import tqdm

DATASET_DIR = "../dataset"

def clean_old_dataset():
    """Removes existing dataset folders to ensure a clean slate."""
    for split in ['train', 'val']:
        split_path = os.path.join(DATASET_DIR, split)
        if os.path.exists(split_path):
            print(f"Cleaning old dataset directory: {split_path}")
            shutil.rmtree(split_path)
        os.makedirs(split_path, exist_ok=True)

def extract_food101():
    clean_old_dataset()
    print("--- Downloading & Processing Pure Food-101 Dataset ---")
    
    # Load dataset from Hugging Face
    ds = load_dataset("ethz/food101")
    
    # Map Hugging Face split names to local directory structure
    split_map = {
        'train': 'train',
        'validation': 'val'
    }

    for hf_split, local_split in split_map.items():
        if hf_split not in ds:
            continue
            
        print(f"\nExtracting {hf_split} split into {DATASET_DIR}/{local_split}/...")
        dataset_split = ds[hf_split]
        labels = dataset_split.features['label'].names
        
        for idx, item in enumerate(tqdm(dataset_split, desc=f"Processing {local_split}")):
            # Extract standard class name (101 clean classes)
            raw_label = labels[item['label']].lower().strip().replace(" ", "_")
            
            target_dir = os.path.join(DATASET_DIR, local_split, raw_label)
            os.makedirs(target_dir, exist_ok=True)
            
            # Save as standard RGB image
            img_path = os.path.join(target_dir, f"food101_{idx}.jpg")
            item['image'].convert("RGB").save(img_path)

    print("\nPure Food-101 dataset extraction complete! 101 clean classes ready.")

if __name__ == "__main__":
    extract_food101()