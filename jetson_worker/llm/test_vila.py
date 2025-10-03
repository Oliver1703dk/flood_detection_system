"""Test VILA1.5-3B inference on Jetson."""

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import config

# Set to use local LLM
config.USE_LOCAL_LLM = True
LOCAL_MODEL_DIR = Path(__file__).resolve().parent / "models" / "Efficient-Large-Model" / "VILA1.5-3b"
config.LOCAL_LLM_MODEL = str(LOCAL_MODEL_DIR)
config.LOCAL_LLM_USE_4BIT = True

from jetson_worker.llm.llm_factory import create_llm_classifier


def test_vila():
    """Test VILA model loading and inference."""
    
    print("=" * 60)
    print("VILA1.5-3B Test on Jetson")
    print("=" * 60)
    
    # Check CUDA
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Create classifier
    print("\n" + "=" * 60)
    print("Creating Local VLM Classifier")
    print("=" * 60)
    
    start_time = time.time()
    classifier = create_llm_classifier(use_local=True)
    load_time = time.time() - start_time
    
    print(f"\n✓ Classifier created in {load_time:.2f} seconds")
    
    # Create a test image
    print("\n" + "=" * 60)
    print("Running Test Inference")
    print("=" * 60)
    
    from PIL import Image
    import io
    
    # Create a simple test image (blue = water-like)
    test_image = Image.new('RGB', (640, 480), color=(100, 150, 200))
    
    # Convert to bytes
    img_byte_arr = io.BytesIO()
    test_image.save(img_byte_arr, format='JPEG')
    image_bytes = img_byte_arr.getvalue()
    
    # Test sensor data
    sensor_data = {
        "temperature": 25.5,
        "humidity": 85.0,
        "pressure": 1010.0,
    }
    sensor_baseline = {
        "temperature": 22.0,
        "humidity": 60.0,
        "pressure": 1013.0,
    }
    
    print("\nRunning inference with test image and sensor data...")
    start_time = time.time()
    
    result = classifier.classify_flood(
        image_bytes,
        sensor_data=sensor_data,
        sensor_baseline=sensor_baseline,
    )
    
    inference_time = time.time() - start_time
    
    print(f"\n✓ Inference completed in {inference_time:.2f} seconds")
    print(f"Result: {result} ({['no-flood', 'little-flood', 'flood'][result]})")
    
    # Memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"\nGPU Memory Usage:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_vila()