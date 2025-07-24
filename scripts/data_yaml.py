import os

# Output path for the YAML file
yaml_path = 'data/yolo_dataset/data.yaml'

# Found from your script
class_indices = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38,
    39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
    57, 58, 59, 60, 61, 62, 63, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76,
    77, 79
]

# Dummy class names for now
class_names = [f'class_{i}' for i in class_indices]

# Compose the YAML content
yaml_content = f"""train: data/yolo_dataset/images/train
val: data/yolo_dataset/images/val

nc: {len(class_names)}
names: {class_names}
"""

# Save it
os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
with open(yaml_path, 'w') as f:
    f.write(yaml_content)

print(f"✅ data.yaml written to: {yaml_path}")
