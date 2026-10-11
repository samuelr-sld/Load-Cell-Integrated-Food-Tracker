import os
import cv2

# Paths relative to execution directory
YOLO_BASE_DIR = "../data"
MOBILENET_DATASET_DIR = "../../mobilenet_pipeline/dataset"

CLASS_MAP = {
    0: "candy", 1: "egg_tart", 2: "french_fries", 3: "chocolate", 4: "biscuit",
    5: "popcorn", 6: "pudding", 7: "ice_cream", 8: "cheese_butter", 9: "cake",
    10: "wine", 11: "milkshake", 12: "coffee", 13: "juice", 14: "milk",
    15: "tea", 16: "almond", 17: "red_beans", 18: "cashew", 19: "dried_cranberries",
    20: "soy", 21: "walnut", 22: "peanut", 23: "egg", 24: "apple", 25: "date",
    26: "apricot", 27: "avocado", 28: "banana", 29: "strawberry", 30: "cherry",
    31: "blueberry", 32: "raspberry", 33: "mango", 34: "olives", 35: "peach",
    36: "lemon", 37: "pear", 38: "fig", 39: "pineapple", 40: "grape", 41: "kiwi",
    42: "melon", 43: "orange", 44: "watermelon", 45: "steak", 46: "pork",
    47: "chicken_duck", 48: "sausage", 49: "fried_meat", 50: "lamb", 51: "sauce",
    52: "crab", 53: "fish", 54: "shellfish", 55: "shrimp", 56: "soup", 57: "bread",
    58: "corn", 59: "hamburg", 60: "pizza", 61: "hanamaki_baozi", 62: "wonton_dumplings",
    63: "pasta", 64: "noodles", 65: "rice", 66: "pie", 67: "tofu", 68: "eggplant",
    69: "potato", 70: "garlic", 71: "cauliflower", 72: "tomato", 73: "kelp",
    74: "seaweed", 75: "spring_onion", 76: "rape", 77: "ginger", 78: "okra",
    79: "lettuce", 80: "pumpkin", 81: "cucumber", 82: "white_radish", 83: "carrot",
    84: "asparagus", 85: "bamboo_shoots", 86: "broccoli", 87: "celery_stick",
    88: "cilantro_mint", 89: "snow_peas", 90: "cabbage", 91: "bean_sprouts",
    92: "onion", 93: "pepper", 94: "green_beans", 95: "french_beans",
    96: "king_oyster_mushroom", 97: "shiitake", 98: "enoki_mushroom",
    99: "oyster_mushroom", 100: "white_button_mushroom", 101: "salad",
    102: "other_ingredients"
}

def extract_crops_from_yolo_labels():
    splits = ['train', 'val']

    for split in splits:
        images_dir = os.path.join(YOLO_BASE_DIR, "images", split)
        labels_dir = os.path.join(YOLO_BASE_DIR, "labels", split)

        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"Skipping {split}: Directory not found at {images_dir}")
            continue

        print(f"Extracting ingredient crops from local YOLO {split} set...")

        for img_name in os.listdir(images_dir):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(images_dir, img_name)
            base_name = os.path.splitext(img_name)[0]
            label_path = os.path.join(labels_dir, f"{base_name}.txt")

            if not os.path.exists(label_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w, _ = img.shape

            with open(label_path, 'r') as f:
                lines = f.readlines()

            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if not parts:
                    continue

                class_id = int(parts[0])
                class_name = CLASS_MAP.get(class_id, f"class_{class_id}")
                coords = [float(x) for x in parts[1:]]

                # Bounding box format: 4 floats [x_center, y_center, bbox_w, bbox_h]
                if len(coords) == 4:
                    x_center, y_center, bw, bh = coords
                    x1 = int((x_center - bw / 2) * w)
                    y1 = int((y_center - bh / 2) * h)
                    x2 = int((x_center + bw / 2) * w)
                    y2 = int((y_center + bh / 2) * h)
                
                # Polygon format: x1, y1, x2, y2...
                elif len(coords) >= 6:
                    # Truncate to an even number of coordinates if odd
                    if len(coords) % 2 != 0:
                        coords = coords[:-1]

                    # Extract pairs safely
                    x_coords = [int(coords[i] * w) for i in range(0, len(coords), 2)]
                    y_coords = [int(coords[i + 1] * h) for i in range(0, len(coords), 2)]

                    if not x_coords or not y_coords:
                        continue

                    x1, x2 = min(x_coords), max(x_coords)
                    y1, y2 = min(y_coords), max(y_coords)
                else:
                    continue

                # Clamp coordinates within bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                crop = img[y1:y2, x1:x2]

                if crop.size > 0 and crop.shape[0] > 10 and crop.shape[1] > 10:
                    target_dir = os.path.join(MOBILENET_DATASET_DIR, split, class_name)
                    os.makedirs(target_dir, exist_ok=True)

                    crop_filename = f"local_foodseg_{base_name}_crop_{idx}.jpg"
                    cv2.imwrite(os.path.join(target_dir, crop_filename), crop)

    print("Local FoodSeg103 ingredient extraction complete!")

if __name__ == "__main__":
    extract_crops_from_yolo_labels()