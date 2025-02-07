import os
import json
import base64
import cv2
import numpy as np

# Import the modules from your project.
# Adjust the import paths as needed based on your project structure.
from cluster_data_receiver.receiver.mqtt_receiver import MQTTReceiver
from cluster_data_receiver.validation.data_validator import DataValidator
from cluster_data_receiver.storage.storage_manager import StorageManager
from yolov8_processor.inference.yolov8_inference import YOLOv8Inference
from yolov8_processor.preprocessing.image_processor import ImageProcessor
from yolov8_processor.postprocessing.result_formatter import ResultFormatter
from flood_classifier.preprocessing.input_validator import InputValidator
from flood_classifier.preprocessing.feature_extractor import FeatureExtractor
from flood_classifier.postprocessing.classification_formatter import ClassificationFormatter
from flood_classifier.model.model_loader import ModelLoader
from flood_classifier.inference.fusion_strategy import FusionStrategy
from flood_classifier.inference.image_classifier import ImageClassifier
from flood_classifier.inference.sensor_classifier import SensorClassifier


def main():
    # ---------------------------
    # 1. Simulate Incoming Data
    # ---------------------------
    # Create a dummy image (a white 640x640 image) and encode it as a Base64 string.
    dummy_image = np.ones((640, 640, 3), dtype=np.uint8) * 255  # White image
    success, buffer = cv2.imencode('.jpg', dummy_image)
    if not success:
        print("Failed to encode dummy image.")
        return
    base64_image = base64.b64encode(buffer).decode('utf-8')

    # Construct a sample data message matching your expected schema.
    sample_message = {
        "image_data": base64_image,
        "sensor_data": {
            "temperature": 25.0,
            "humidity": 50.0,
            "pressure": 1013.25
        },
        "metadata": {
            "timestamp": "2025-02-07T12:00:00Z",
            "location": "Test Location",
            "camera_id": "CAM123"
        }
    }

    print("\n--- Sample Message JSON ---")
    print(json.dumps(sample_message, indent=4))

    # ----------------------------------------------------
    # 2. Validate the Data & Store It if Validation Passes
    # ----------------------------------------------------
    validator = DataValidator()
    storage_manager = StorageManager()

    if validator.validate(sample_message):
        storage_manager.store(sample_message)
        print("Data validated and stored successfully.")
    else:
        print("Data validation failed. Exiting simulation.")
        return

    # -------------------------------------------
    # 3. Process the Image and Run YOLOv8 Inference
    # -------------------------------------------
    # Preprocess the Base64-encoded image (decode, resize, normalize)
    image_processor = ImageProcessor()
    preprocessed_image = image_processor.preprocess(sample_message["image_data"])
    print("Image preprocessed for inference.")

    # Run YOLOv8 inference on the preprocessed image.
    # (Make sure the YOLOv8 model file exists at the specified path.)
    try:
        inference = YOLOv8Inference()
        # The inference method may expect an image array.
        results = inference.run_inference(preprocessed_image)
    except Exception as e:
        print(f"Error during YOLOv8 inference: {e}")
        results = None

    # Format the detection results using your result formatter.
    result_formatter = ResultFormatter(confidence_threshold=0.5)
    if results is not None:
        detection_results = result_formatter.format_results(results)
    else:
        detection_results = []
    print("\n--- YOLOv8 Detection Results ---")
    print(detection_results)

    # -------------------------------------------------
    # 4. Flood Classification via Sensor & Image Data
    # -------------------------------------------------
    # (A) Sensor-based Classification
    model_loader = ModelLoader()
    try:
        sensor_model = model_loader.load_sensor_model()
        sensor_classifier = SensorClassifier()
        sensor_pred = sensor_classifier.predict(sensor_model, sample_message["sensor_data"])
        print(f"Sensor-based flood prediction (0: No Flood, 1: Some Water, 2: Flooded): {sensor_pred}")
    except Exception as e:
        print(f"Error during sensor classification: {e}")
        sensor_pred = None

    # (B) Image-based Classification using YOLO detections
    image_classifier = ImageClassifier()
    image_pred = image_classifier.classify_flood(detection_results)
    print(f"Image-based flood prediction (0: No Flood, 1: Some Water, 2: Flooded): {image_pred}")

    # (C) Fuse the two predictions
    fusion = FusionStrategy()
    final_prediction = fusion.merge_predictions(sensor_pred, image_pred)

    # Format the final classification result for display.
    classification_formatter = ClassificationFormatter()
    final_result = classification_formatter.format_output(final_prediction)
    print("\n--- Final Flood Classification ---")
    print(final_result)

    # -------------------------------------------------
    # 5. (Optional) Start the MQTT Receiver to listen for data.
    # -------------------------------------------------
    # Uncomment the following lines if you wish to run the MQTT receiver.
    #
    # receiver = MQTTReceiver(
    #     broker_url="mqtt.example.com",  # Replace with your MQTT broker URL
    #     broker_port=1883,               # Replace with your MQTT broker port if different
    #     topic="your/topic/here"
    # )
    # receiver.start()


if __name__ == "__main__":
    main()
