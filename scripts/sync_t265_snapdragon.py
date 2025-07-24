"""
sync_t265_snapdragon.py

Synchronizes RGB camera images with IMU data for each sequence in your SLAM dataset.
Expects each sequence folder (e.g. seq_01) in the input folder to contain:
  - A subfolder "rgb/" with images (filenames should be timestamps, e.g., 1629381234.567890.png)
  - An IMU file "imu.txt" (CSV format with a header: timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z)

The script creates matching subfolders in the output directory and writes:
  - A "rgb/" folder with the image files copied over.
  - A "synced_imu.txt" file in CSV format that pairs each image timestamp with the closest IMU data.
"""

import os
import argparse
import csv
import glob
import shutil

def parse_args():
    parser = argparse.ArgumentParser(
        description="Synchronize RGB images with IMU data using a specified time tolerance."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input directory containing split sequences (e.g., data/slam/splits_track/)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output directory for synchronized data (e.g., data/slam/synced_data/)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Time tolerance (in seconds) to match IMU data to image timestamps (default: 0.01)",
    )
    return parser.parse_args()

def load_imu_data(imu_file):
    """
    Reads the imu.txt file and returns a list of IMU records.
    Each record is a dictionary with keys: timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z.
    """
    imu_data = []
    with open(imu_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                imu_data.append({
                    'timestamp': float(row['timestamp']),
                    'acc_x': row['acc_x'],
                    'acc_y': row['acc_y'],
                    'acc_z': row['acc_z'],
                    'gyro_x': row['gyro_x'],
                    'gyro_y': row['gyro_y'],
                    'gyro_z': row['gyro_z']
                })
            except Exception as e:
                print(f"Error parsing IMU row {row}: {e}")
    return imu_data

def sync_sequence(seq_path, output_seq_path, tolerance):
    """
    Synchronizes image data with IMU data for one sequence.
    - Copies images from seq_path/rgb to output_seq_path/rgb.
    - Finds the closest IMU measurement for each image timestamp.
    - Writes a CSV file with the synchronized IMU data.
    """
    rgb_dir = os.path.join(seq_path, "rgb")
    imu_file = os.path.join(seq_path, "imu.txt")

    if not os.path.exists(rgb_dir) or not os.path.exists(imu_file):
        print(f"Skipping {seq_path}: Missing 'rgb' folder or 'imu.txt' file.")
        return

    imu_data = load_imu_data(imu_file)
    # Ensure the IMU data is sorted by timestamp
    imu_data.sort(key=lambda x: x['timestamp'])

    output_rgb_dir = os.path.join(output_seq_path, "rgb")
    os.makedirs(output_rgb_dir, exist_ok=True)

    synced_imu = []  # To store synchronized IMU entries

    # Get list of image files (assuming .png extension)
    image_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    for img_path in image_files:
        # Extract timestamp from filename.
        # Assumes filename pattern: "<timestamp>.png"
        basename = os.path.basename(img_path)
        timestamp_str, _ = os.path.splitext(basename)
        try:
            img_ts = float(timestamp_str)
        except ValueError:
            print(f"Error: Could not parse timestamp from filename {basename}. Skipping this file.")
            continue

        # Find the closest IMU record for the image timestamp
        closest_imu = min(imu_data, key=lambda imu: abs(imu['timestamp'] - img_ts))
        if abs(closest_imu['timestamp'] - img_ts) <= tolerance:
            # Copy the image file to the output directory
            shutil.copy2(img_path, output_rgb_dir)
            # Record the synchronized data
            synced_imu.append({"image_timestamp": img_ts, **closest_imu})
        else:
            print(f"No matching IMU data within tolerance for image {basename}")

    # Write the synchronized IMU data to a CSV file
    output_imu_file = os.path.join(output_seq_path, "synced_imu.txt")
    with open(output_imu_file, 'w', newline='') as f_out:
        fieldnames = ["image_timestamp", "timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for entry in synced_imu:
            writer.writerow(entry)
    print(f"Sequence synchronized and saved in: {output_seq_path}")

def main():
    args = parse_args()

    input_dir = args.input
    output_dir = args.output
    tolerance = args.tolerance

    os.makedirs(output_dir, exist_ok=True)

    # Process each sequence folder in the input directory.
    # Assumes sequence folders are named like "seq_01", "seq_02", etc.
    for seq in os.listdir(input_dir):
        seq_path = os.path.join(input_dir, seq)
        if os.path.isdir(seq_path) and seq.startswith("seq_"):
            output_seq_path = os.path.join(output_dir, seq)
            os.makedirs(output_seq_path, exist_ok=True)
            print(f"Synchronizing sequence: {seq}")
            sync_sequence(seq_path, output_seq_path, tolerance)
    print("Data synchronization complete.")

if __name__ == "__main__":
    main()
