"""
Builds data.yaml for YOLO training from FoodSeg103's category_id.txt,
which is formatted as: <id>\t<name>  (one entry per line)

Usage:
    python build_data_yaml.py
"""

names = {}
with open("category_id.txt", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        idx_str, name = line.split("\t", maxsplit=1)
        names[int(idx_str)] = name.strip()

print(f"Parsed {len(names)} categories")
print("Sample entries:", list(names.items())[:5])

with open("data.yaml", "w", encoding="utf-8") as f:
    f.write("path: ../yolo_pipeline/data\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("names:\n")
    for idx in sorted(names):
        # FoodSeg103 category ids start at 1 (0 = background, excluded from masks/labels)
        # YOLO needs 0-indexed classes, so subtract 1
        f.write(f"  {idx - 1}: {names[idx]}\n")

print("Wrote data.yaml")