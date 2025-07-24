import torch
from ultralytics import YOLO

# Clear unused GPU memory before starting the training
torch.cuda.empty_cache()

# Set the environment variable to avoid memory fragmentation
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main():
    # Load the YOLOv8 model with a lighter configuration (e.g., yolov8s.yaml)
    model = YOLO('yolov8s.yaml')  # Use 'yolov8s.yaml' or 'yolov8n.yaml' for a lighter model

    # Start the training process
    model.train(
        data="data/yolo_dataset/data.yaml",    # Replace with your dataset configuration
        epochs=100,                  # Set the number of epochs
        imgsz=416,                   # Set the image size to 416 for reduced memory usage
        batch=1,                     # Reduce batch size to 1
        workers=2                    # Set number of workers (adjust as needed)
    )

if __name__ == "__main__":
    main()
