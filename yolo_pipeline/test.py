from ultralytics.data.utils import visualize_image_annotations
import yaml

with open("data.yaml") as f:
    data_cfg = yaml.safe_load(f)

label_map = data_cfg["names"]

visualize_image_annotations(
    "data/images/train/000000.jpg",
    "data/labels/train/000000.txt",
    label_map=label_map
)