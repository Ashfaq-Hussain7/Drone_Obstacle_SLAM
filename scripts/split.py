import os
import random
import shutil
from pathlib import Path

# Set base dataset directory correctly
dataset_path = Path("C:/Users/ashfa/OneDrive/Desktop/Drone_Obstacle_SLAM/data/yolo_dataset")

# Define subdirectories
image_train_dir = dataset_path / "C:/Users/ashfa/OneDrive/Desktop/Drone_Obstacle_SLAM/data/yolo_dataset/images/train"
label_train_dir = dataset_path / "C:/Users/ashfa/OneDrive/Desktop/Drone_Obstacle_SLAM/data/yolo_dataset/labels/train"
image_val_dir = dataset_path / "C:/Users/ashfa/OneDrive/Desktop/Drone_Obstacle_SLAM/data/yolo_dataset/images/val"
label_val_dir = dataset_path / "C:/Users/ashfa/OneDrive/Desktop/Drone_Obstacle_SLAM/data/yolo_dataset/labels/val"

# Create validation folders if they don't exist
image_val_dir.mkdir(parents=True, exist_ok=True)
label_val_dir.mkdir(parents=True, exist_ok=True)

# Get all image files from training directory
image_files = list(image_train_dir.glob("*.png"))  # or *.jpg
random.shuffle(image_files)

# Perform 20% split
val_count = int(0.2 * len(image_files))
val_images = image_files[:val_count]

# Move images and corresponding labels to val folders
moved_count = 0
for img_path in val_images:
    label_path = label_train_dir / f"{img_path.stem}.txt"

    if label_path.exists():
        shutil.move(str(img_path), image_val_dir / img_path.name)
        shutil.move(str(label_path), label_val_dir / label_path.name)
        moved_count += 1

print(f"✅ Moved {moved_count} images and their labels to validation set.")
