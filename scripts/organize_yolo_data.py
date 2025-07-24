import os
import shutil
from tqdm import tqdm

# Define paths
ANNOTATION_BATCH_DIR = 'data/annotations/train'
YOLO_IMAGES_DIR = 'data/yolo_dataset/images/train'
YOLO_LABELS_DIR = 'data/yolo_dataset/labels/train'

# Create target folders if not exist
os.makedirs(YOLO_IMAGES_DIR, exist_ok=True)
os.makedirs(YOLO_LABELS_DIR, exist_ok=True)

# Process each batch folder
batch_folders = [f for f in os.listdir(ANNOTATION_BATCH_DIR) if os.path.isdir(os.path.join(ANNOTATION_BATCH_DIR, f))]

print(f"Found {len(batch_folders)} batches. Flattening now...")

for batch in tqdm(batch_folders):
    batch_path = os.path.join(ANNOTATION_BATCH_DIR, batch)

    for file in os.listdir(batch_path):
        full_path = os.path.join(batch_path, file)

        if file.endswith('.png') and not file.endswith('_mask.png'):
            # This is an image file
            shutil.copy(full_path, os.path.join(YOLO_IMAGES_DIR, file))
        
        elif file.endswith('.txt'):
            # This is a YOLO label file
            shutil.copy(full_path, os.path.join(YOLO_LABELS_DIR, file))

print("✅ Dataset flattened and organized for YOLOv8 training.")
