import os
import json
import base64
import cv2
import numpy as np

# Import modules from your project.
from cluster_data_receiver.receiver.mqtt_receiver import MQTTReceiver
from cluster_data_receiver.validation.data_validator import DataValidator
from cluster_data_receiver.storage.storage_manager import StorageManager
from flood_classifier.baselinecalculator.baseline_calculator import BaselineCalculator
from flood_classifier.postprocessing.data_result_saver import DataResultsSaver
from yolov8_processor.inference.multi_model_inference import MultiModelInference
from yolov8_processor.inference.yolov8_inference import YOLOv8Inference
from yolov8_processor.preprocessing.image_processor import ImageProcessor
from yolov8_processor.postprocessing.result_formatter import ResultFormatter
from flood_classifier.preprocessing.input_validator import InputValidator
from flood_classifier.preprocessing.feature_extractor import FeatureExtractor
from flood_classifier.postprocessing.classification_formatter import ClassificationFormatter
from flood_classifier.model.model_loader import ModelLoader
from flood_classifier.inference.fusion_strategy import FusionStrategy
from flood_classifier.inference.image_classifier import ImageClassifier
from flood_classifier.inference.image_classifier_2 import EnhancedImageClassifier
from flood_classifier.inference.sensor_classifier import SensorClassifier
# Import the CombinedClassifier for one-step classification.
from flood_classifier.inference.combined_classifier import CombinedClassifier


# Import configuration
import config


def simulate_message():
    """
    Simulates an incoming data message.
    Creates a dummy white image, encodes it to Base64,
    and constructs a sample data dictionary.
    """
    

    if(config.IMAGE_MODE == "test"): 
        print(f'Using test image {config.IMAGE_NAME}')
        # Construct the test image file path.
        test_image_path = os.path.join("test_images", config.IMAGE_NAME +'.jpg')
        
        # Read the image from disk.
        image = cv2.imread(test_image_path)
        if image is None:
            raise ValueError(f"Failed to load test image from path: {test_image_path}")
        print('Image loaded successfully')
    else: 
        # Create a dummy white 640x640 image.
        image = np.ones((config.IMAGE_SIZE[1], config.IMAGE_SIZE[0], 3), dtype=np.uint8) * 255


    # Encode the image as JPEG
    success, buffer = cv2.imencode('.jpg', image)
    if not success:
        raise ValueError("Failed to encode dummy image.")
    
    # Convert the image to a Base64 string
    base64_image = base64.b64encode(buffer).decode('utf-8')

    # Construct sample message.
    message = {
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
    return message


def main():
    # -------------------------------
    # Choose Classification Mode:
    # -------------------------------
    # "combined" uses the CombinedClassifier.
    # "fused" uses separate sensor & image classifiers with fusion.
    classification_mode = config.CLASSIFICATION_MODE  # "fused" or "combined"

    # -------------------------------------------
    # 1. Simulate Incoming Data
    # -------------------------------------------
    try:
        sample_message = simulate_message()
        print("\n--- Simulated Message ---")
        # print(json.dumps(sample_message, indent=4))
    except Exception as e:
        print("Error simulating message:", e)
        return

    # -------------------------------------------
    # 2. Validate and Store Data
    # -------------------------------------------
    validator = DataValidator()
    storage_manager = StorageManager()
    if validator.validate(sample_message):
        storage_manager.store(sample_message)
        print("Data validated and stored successfully.")
    else:
        print("Data validation failed. Exiting simulation.")
        return

    # -------------------------------------------
    # 3. Image Preprocessing & YOLOv8 Inference
    # -------------------------------------------
    image_processor = ImageProcessor()
    try:
        preprocessed_image = image_processor.preprocess(sample_message["image_data"])
        print("Image preprocessed for inference.")
    except Exception as e:
        print("Error in image preprocessing:", e)
        return
    
    

    try:
        # inference = YOLOv8Inference()
        # results = inference.run_inference(preprocessed_image)
        # New multi-model call:
        model_info = [
            ("1", "best1.pt"),
            ("2", "best2.pt"),
            ("3", "best3.pt"),
            ("4", "best4.pt"),
            ("5", "best5.pt"),
        ]
        multi_inference = MultiModelInference(model_info)
        aggregated_results = multi_inference.run_all_inference(preprocessed_image)

        print("YOLOv8 inference completed.")
    except Exception as e:
        print("Error during YOLOv8 inference:", e)
        results = None

    # Format the YOLO detection results.
    result_formatter = ResultFormatter()
    detection_results = result_formatter.format_results(aggregated_results) if aggregated_results else []
    print("\n--- YOLOv8 Detection Results ---")
    print(detection_results)

    # -------------------------------------------
    # 4. Flood Classification
    # -------------------------------------------
    classification_formatter = ClassificationFormatter()

    if classification_mode == "combined":
        # Use the CombinedClassifier to fuse image and sensor data internally.
        combined_classifier = CombinedClassifier(
            image_weight=1.0, sensor_weight=1.0,
            threshold_low=0.2, threshold_high=0.5
        )
        # Build an input in the expected format.
        combined_input = {
            "feature_vector": {
                "image_data": detection_results,  # List of detections (each with 'bounding_box' and 'confidence').
                "sensor_data": [
                    sample_message["sensor_data"]["temperature"],
                    sample_message["sensor_data"]["humidity"],
                    sample_message["sensor_data"]["pressure"]
                ]
            }
        }
        # Note: Adjust the image_size if necessary. Here, we use (640, 640) matching our dummy image.
        combined_pred = combined_classifier.classify(combined_input, image_size=config.IMAGE_SIZE)
        print("Combined Classifier prediction (0: No Flood, 1: Some Water, 2: Flooded):", combined_pred)
        final_result = classification_formatter.format_output(combined_pred)
    elif classification_mode == "fused":
        # (A) Sensor-based Classification.
        try:
            model_loader = ModelLoader()
            sensor_model = model_loader.load_sensor_model()
            sensor_classifier = SensorClassifier()
            sensor_pred = sensor_classifier.predict(sensor_model, sample_message["sensor_data"])
            print("Sensor-based flood prediction (0: No Flood, 1: Some Water, 2: Flooded):", sensor_pred)
        except Exception as e:
            print("Error during sensor classification:", e)
            sensor_pred = None

        # (B) Image-based Classification using YOLO detections.
        # image_classifier = ImageClassifier()
        image_classifier = EnhancedImageClassifier()
        image_pred = image_classifier.classify_flood(detection_results)
        print("Image-based flood prediction (0: No Flood, 1: Some Water, 2: Flooded):", image_pred)

        # (C) Fuse the Two Predictions.
        fusion = FusionStrategy()
        final_pred = fusion.merge_predictions(sensor_pred, image_pred)
        final_result = classification_formatter.format_output(final_pred)
    else:
        print("Invalid classification mode selected.")
        return
    
    # Save the data and classification results.
    saver = DataResultsSaver()
    saver.save(sample_message, final_result)

    # -------------------------------------------
    # 5. Update the Baseline Using Latest Data
    # -------------------------------------------
    baseline_calculator = BaselineCalculator(
        results_dir="storage/data_results",
        baseline_file="storage/sensor_baselines.json",
        tau=12
    )

    print("\n🔄 Updating sensor baselines...")
    current_baselines = baseline_calculator.update_baselines()

    if current_baselines:
        print("✅ Updated Baselines:", current_baselines)
    else:
        print("❌ Baseline update failed (no stable period found).")

    print("\n--- Final Flood Classification ---")
    print(final_result)

    # -------------------------------------------
    # 5. (Optional) Start MQTT Receiver
    # -------------------------------------------
    # Uncomment the following lines to start the MQTT receiver.
    #
    # receiver = MQTTReceiver(
    #     broker_url=config.MQTT_BROKER_URL,  # Replace with your MQTT broker URL.
    #     broker_port=config.MQTT_BROKER_PORT,               # Replace with your MQTT broker port if different.
    #     topic=config.MQTT_TOPIC
    # )
    # receiver.start()


if __name__ == "__main__":
    main()
