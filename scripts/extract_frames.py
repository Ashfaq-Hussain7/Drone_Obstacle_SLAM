import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Define source and destination paths
ENVIRONMENTS = ["indoor_forward", "indoor_downward", "outdoor_forward", "outdoor_downward"]
SOURCE_ROOT = "C:/Users/ashfa/OneDrive/Desktop/Drone_Obstacle_SLAM/data/obstacle_detection"
DESTINATION = "C:/Users/ashfa/OneDrive/Desktop/Drone_Obstacle_SLAM/data/images/train"

# Create destination folder if it doesn't exist
os.makedirs(DESTINATION, exist_ok=True)

def extract_frames():
    for env in ENVIRONMENTS:
        env_path = Path(SOURCE_ROOT) / env
        if not env_path.exists():
            print(f"[WARN] Environment folder not found: {env_path}")
            continue

        for seq_folder in sorted(env_path.iterdir()):
            if seq_folder.is_dir():
                img_folder = seq_folder / "img"
                if not img_folder.exists():
                    print(f"[WARN] No img folder found in {seq_folder}")
                    continue
                
                for img_file in tqdm(sorted(img_folder.glob("*.*")), desc=f"Extracting {env}/{seq_folder.name}"):
                    if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                        # Compose unique filename: env_seq_imgname.png
                        new_name = f"{env}_{seq_folder.name}_{img_file.name}"
                        dest_path = Path(DESTINATION) / new_name
                        shutil.copy(img_file, dest_path)

    print(f"\n✅ Frame extraction complete. Images saved to: {DESTINATION}")

if __name__ == "__main__":
    extract_frames()
