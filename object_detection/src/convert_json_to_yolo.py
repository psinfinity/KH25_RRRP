import os
import json

# Paths
json_path = "\give path to ur json directory"
image_dir = r"data\images\train"
label_dir = r"data\labels\train"

# Ensure label directory exists
os.makedirs(label_dir, exist_ok=True)

# Class mapping (excluding "drivable area" and "lane")
class_map = {
    "car": 0, "bus": 1, "truck": 2, "person": 3,
    "rider": 4, "bike": 5, "motor": 6, "traffic light": 7,
    "traffic sign": 8, "train": 9
}

# Load JSON file
with open(json_path, "r") as file:
    data = json.load(file)

# Process each image annotation
for item in data:
    image_name = item["name"]
    label_file = os.path.join(label_dir, image_name.replace(".jpg", ".txt"))

    with open(label_file, "w") as f:
        for obj in item["labels"]:
            category = obj["category"]
            if category not in class_map:
                continue  # Skip drivable area and lane

            class_id = class_map[category]
            box = obj.get("box2d", None)
            if not box:
                continue  # Skip objects without bounding boxes
            
            # Normalize YOLO format
            x_center = ((box["x1"] + box["x2"]) / 2) / 1280  # Assuming width = 1280
            y_center = ((box["y1"] + box["y2"]) / 2) / 720   # Assuming height = 720
            width = (box["x2"] - box["x1"]) / 1280
            height = (box["y2"] - box["y1"]) / 720

            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

print("✅ Conversion complete! Check your YOLO label files in 'labels/val/'.")
