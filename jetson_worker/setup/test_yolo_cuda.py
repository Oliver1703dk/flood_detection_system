import torch
from ultralytics import YOLO
import config

print(f"CUDA available: {torch.cuda.is_available()}")

# Load a YOLO model
model_path = "yolov8_processor/model/nano/best1.pt"  # Adjust path
model = YOLO(model_path)

# Check model device
print(f"Model device: {next(model.model.parameters()).device}")

# Run a test inference
import numpy as np
test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

print("Running inference...")
results = model(test_image, device='cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ Inference completed successfully!")