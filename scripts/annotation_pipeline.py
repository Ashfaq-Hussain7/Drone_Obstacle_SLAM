import os
import shutil
import json
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import set_start_method, Manager, Value, Queue
import multiprocessing
from math import ceil
import time
import sys
from datetime import timedelta
from tqdm import tqdm


# Make sure to set the multiprocessing start method to 'spawn' for CUDA compatibility
try:
    set_start_method('spawn')
except RuntimeError:
    # Method already set
    pass

# === Configs ===
YOLO_MODEL_PATH = 'models/yolov8x.pt'
SAM_CHECKPOINT = "models/sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"
# Set device to CUDA if available, otherwise use CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Lower confidence threshold for detecting more objects
CONFIDENCE_THRESHOLD = 0.1
# Enable debug mode to see what's happening during detection
DEBUG_MODE = True

# === Utility Functions ===
def save_image(image, path):
    Image.fromarray(image).save(path)

def save_mask(mask, path):
    mask = np.squeeze(mask)
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)
    Image.fromarray(mask).save(path)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def save_txt(bboxes, path):
    with open(path, 'w') as f:
        for bbox in bboxes:
            f.write(" ".join(map(str, bbox)) + "\n")

# Simple progress bar fallback if tqdm is not available
class SimpleProgressBar:
    def __init__(self, total, desc=None, unit='it'):
        self.total = total
        self.desc = desc or ''
        self.unit = unit
        self.n = 0
        self.start_time = time.time()
        self._print_progress()

    def update(self, n=1):
        self.n += n
        self._print_progress()

    def _print_progress(self):
        percent = 100 * (self.n / self.total)
        elapsed = time.time() - self.start_time

        # Calculate ETA
        if self.n > 0:
            eta = elapsed * (self.total - self.n) / self.n
            eta_str = str(timedelta(seconds=int(eta)))
        else:
            eta_str = "Unknown"

        bar_length = 40
        filled_length = int(bar_length * self.n // self.total)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)

        sys.stdout.write(f"\r{self.desc} |{bar}| {percent:.1f}% ({self.n}/{self.total}) ETA: {eta_str}")
        sys.stdout.flush()

        if self.n >= self.total:
            sys.stdout.write('\n')

    def close(self):
        if self.n < self.total:
            self.n = self.total
            self._print_progress()

# === Core Annotation Logic for One Batch ===
def process_batch(batch_id, image_paths, output_root, progress_queue=None):
    try:
        # Load YOLO model with GPU acceleration
        detector = YOLO(YOLO_MODEL_PATH)
        detector.to(DEVICE)

        # Load SAM model with GPU acceleration
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
        sam.to(DEVICE)
        sam_predictor = SamPredictor(sam)

        batch_folder = os.path.join(output_root, f"batch_{batch_id}")
        os.makedirs(batch_folder, exist_ok=True)

        # Initialize progress tracking for this batch
        total_in_batch = len(image_paths)

        # Counter for debug info
        empty_detections = 0
        successful_detections = 0

        for i, img_path in enumerate(image_paths):
            try:
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Use GPU for YOLO prediction with lower confidence threshold
                results = detector.predict(img, save=False, conf=CONFIDENCE_THRESHOLD, device=DEVICE)[0]
                bboxes = []
                masks = []
                meta = {"boxes": []}

                h, w = img.shape[:2]

                # Debug output for detection results
                if DEBUG_MODE and i % 100 == 0:
                    print(f"\n[Batch {batch_id}] Image {i}/{total_in_batch}: {base_name}")
                    print(f"Detected {len(results.boxes)} boxes with confidence threshold {CONFIDENCE_THRESHOLD}")

                # Check if there are detections
                if len(results.boxes) == 0:
                    empty_detections += 1
                    # Create a fallback box covering the entire image if no detections
                    if DEBUG_MODE and i % 100 == 0:
                        print(f"No detections found. Using fallback whole-image box.")

                    # Add a fallback box that covers the entire image
                    xyxy = [0, 0, w, h]

                    x_center = 0.5  # Center of image
                    y_center = 0.5  # Center of image
                    bbox_width = 1.0  # Full width
                    bbox_height = 1.0  # Full height

                    # Use class 0 (usually "person" or "object" in most models) with low confidence
                    bboxes.append([0, x_center, y_center, bbox_width, bbox_height])
                    meta["boxes"].append({"class": 0, "conf": 0.5, "xyxy": xyxy})

                    # Process with SAM
                    sam_predictor.set_image(img_rgb)
                    input_box = np.array(xyxy).reshape((1, 4))
                    input_box_tensor = torch.tensor(input_box, device=DEVICE)
                    masks_sam, _, _ = sam_predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=input_box_tensor,
                        multimask_output=False
                    )
                    masks.append(masks_sam[0].cpu().numpy())
                else:
                    successful_detections += 1
                    # Process normal detections
                    for det in results.boxes:
                        cls = int(det.cls.item())
                        conf = float(det.conf.item())
                        xyxy = det.xyxy.cpu().numpy()[0].tolist()

                        x_center = ((xyxy[0] + xyxy[2]) / 2) / w
                        y_center = ((xyxy[1] + xyxy[3]) / 2) / h
                        bbox_width = (xyxy[2] - xyxy[0]) / w
                        bbox_height = (xyxy[3] - xyxy[1]) / h
                        bboxes.append([cls, x_center, y_center, bbox_width, bbox_height])
                        meta["boxes"].append({"class": cls, "conf": conf, "xyxy": xyxy})

                        sam_predictor.set_image(img_rgb)
                        input_box = np.array(xyxy).reshape((1, 4))
                        # Send box tensor to GPU
                        input_box_tensor = torch.tensor(input_box, device=DEVICE)
                        masks_sam, _, _ = sam_predictor.predict_torch(
                            point_coords=None,
                            point_labels=None,
                            boxes=input_box_tensor,
                            multimask_output=False
                        )
                        masks.append(masks_sam[0].cpu().numpy())

                save_txt(bboxes, os.path.join(batch_folder, base_name + ".txt"))
                save_json(meta, os.path.join(batch_folder, base_name + ".json"))
                if masks:
                    combined_mask = np.any(np.stack(masks, axis=0), axis=0)
                    save_mask(combined_mask, os.path.join(batch_folder, base_name + "_mask.png"))
                shutil.copy(img_path, os.path.join(batch_folder, base_name + ".png"))

                # Update progress
                if progress_queue is not None:
                    progress_queue.put(1)  # Increment the shared progress counter

            except Exception as e:
                print(f"\n[Batch {batch_id}] Error in {img_path}: {e}")
                if progress_queue is not None:
                    progress_queue.put(1)  # Still increment counter even if there was an error

        # Print detection stats at the end of each batch
        print(f"\n[Batch {batch_id}] Detection stats: {successful_detections} successful, {empty_detections} empty")
        print(f"[Batch {batch_id}] Detection rate: {successful_detections/(successful_detections+empty_detections)*100:.2f}%")

    except Exception as e:
        print(f"\n[Batch {batch_id}] Fatal error: {e}")
        # If we had a fatal error, increment the queue for all remaining images in this batch
        if progress_queue is not None:
            for _ in range(len(image_paths)):
                progress_queue.put(1)

    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return batch_id

# === Main Function: Safe Parallel with Per-Batch Processing ===
def process_images_in_batches(image_dir, output_dir, batch_size=1000, max_workers=2):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = sorted([
        os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")
    ])
    total_images = len(image_paths)
    total_batches = ceil(total_images / batch_size)

    print(f"Processing {total_images} images in {total_batches} batches on {DEVICE}")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # For GPU processing with spawn method, fewer workers are better
        max_workers = min(max_workers, 2)

    # Setup progress tracking
    manager = Manager()
    progress_queue = manager.Queue()

    # Create batch args
    batch_args = []
    for i in range(total_batches):
        start = i * batch_size
        end = min(start + batch_size, total_images)
        batch_args.append((i, image_paths[start:end], output_dir, progress_queue))

    # Setup the overall progress bar
    pbar = tqdm(total=total_images, desc="Total Progress", unit="img")

    start_time = time.time()
    completed_batches = 0

    # Use the multiprocessing module directly with spawn method
    with ProcessPoolExecutor(mp_context=multiprocessing.get_context('spawn'), max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_batch = {executor.submit(process_batch, *args): args[0] for args in batch_args}

        # Process progress updates
        while completed_batches < total_batches:
            # Check if any batch is complete
            for future in as_completed(future_to_batch, timeout=0.01):
                if future.done():
                    batch_id = future.result()
                    completed_batches += 1
                    print(f"Batch {batch_id} completed ({completed_batches}/{total_batches})")

            # Process any progress updates
            try:
                while True:  # Process all available updates
                    progress_queue.get_nowait()
                    pbar.update(1)
            except:
                pass  # Queue is empty

            # Calculate and display ETA
            if completed_batches > 0:
                elapsed = time.time() - start_time
                images_per_sec = pbar.n / elapsed if elapsed > 0 else 0
                remaining_images = total_images - pbar.n
                eta_seconds = remaining_images / images_per_sec if images_per_sec > 0 else 0
                eta = str(timedelta(seconds=int(eta_seconds)))

                # Print ETA if using SimpleProgressBar (tqdm shows it automatically)
                if not pbar.n % 10 == 0:  # Update less frequently
                    print(f"\nProcessing speed: {images_per_sec:.2f} images/sec, ETA: {eta}")

            time.sleep(0.1)  # Prevent busy waiting

    pbar.close()
    elapsed = time.time() - start_time
    print(f"\nTotal processing time: {timedelta(seconds=int(elapsed))}")
    print(f"Average processing speed: {total_images / elapsed:.2f} images/sec")

# === Entry Point ===
if __name__ == "__main__":
    image_folder = "data/images/train"
    output_folder = "data/annotations/train"
    process_images_in_batches(image_folder, output_folder, batch_size=1000, max_workers=1)  # Reduced workers for GPU
