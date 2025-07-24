from ultralytics import YOLO
import os

# Path to the trained model
model_path = '/kaggle/input/ashfaqs-test/test_image/best.pt'

# Path to the test image (update this path if your image is elsewhere)
test_image_path = '/kaggle/input/ashfaqs-test/test_image/test_image.png'

# Verify that the test image exists
if not os.path.exists(test_image_path):
    raise FileNotFoundError(f"Test image not found at {test_image_path}. Please upload the image to the specified path.")

# Load the trained model
model = YOLO(model_path)

# Perform inference on the test image
results = model.predict(
    source=test_image_path,
    conf=0.01,  # Confidence threshold
    iou=0.45,   # IoU threshold for NMS
    save=True,  # Save the output image with bounding boxes
    save_txt=True,  # Save detection results as a text file
    project='/kaggle/working',  # Output directory
    name='test_inference',      # Name of the inference run
    exist_ok=True              # Allow overwriting existing output
)

# Print detection results
for result in results:
    print("\nDetected objects:")
    for box in result.boxes:
        class_id = int(box.cls)
        class_name = model.names[class_id]
        confidence = box.conf.item()
        print(f"Class: {class_name}, Confidence: {confidence:.2f}, BBox: {box.xywh.tolist()}")

# Path to the saved output image
output_image_path = '/kaggle/working/test_inference/test_image.jpg'
print(f"✅ Inference complete. Output image saved to {output_image_path}")