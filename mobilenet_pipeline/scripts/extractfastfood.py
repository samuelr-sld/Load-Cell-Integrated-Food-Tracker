import os
from datasets import load_dataset

DATASET_DIR = "../dataset"

FASTFOOD_TAXONOMY_MAP = {
    "burger": "hamburg",            
    "french_fries": "french_fries", 
    "fries": "french_fries",
    "pizza": "pizza",               
    "hot_dog": "hot_dog",           
    "donut": "donuts",              
}

def extract_fastfood_bulk():
    print("--- Bulk Downloading & Extracting Fast Food Dataset ---")
    
    # Downloads dataset as a single local batch
    ds = load_dataset("andrewsunanda/fast_food_image_classification")
    
    split_map = {
        'train': 'train',
        'validation': 'val',
        'test': 'val'
    }

    for hf_split, local_split in split_map.items():
        if hf_split not in ds:
            continue
            
        print(f"\nProcessing {hf_split} split...")
        dataset_split = ds[hf_split]
        labels = dataset_split.features['label'].names
        
        for idx, item in enumerate(dataset_split):
            raw_label = str(labels[item['label']]).lower().strip().replace(" ", "_")
            target_class = FASTFOOD_TAXONOMY_MAP.get(raw_label, f"fastfood_{raw_label}")
            
            target_dir = os.path.join(DATASET_DIR, local_split, target_class)
            os.makedirs(target_dir, exist_ok=True)
            
            img_path = os.path.join(target_dir, f"fastfood_{idx}.jpg")
            item['image'].convert("RGB").save(img_path)

            if (idx + 1) % 500 == 0:
                print(f"Saved {idx + 1}/{len(dataset_split)} images to {local_split}...")

    print("\nFast Food extraction complete!")

if __name__ == "__main__":
    extract_fastfood_bulk()