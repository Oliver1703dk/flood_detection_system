from datetime import datetime
import os
import json
import base64
import cv2
import numpy as np

# Import your project modules.
from cluster_data_receiver.receiver.mqtt_receiver import MQTTReceiver
from cluster_data_receiver.validation.data_validator import DataValidator
from cluster_data_receiver.storage.storage_manager import StorageManager
from flood_classifier.baselinecalculator.baseline_calculator import BaselineCalculator
from flood_classifier.inference.classifier_both import ClassifierBoth
from flood_classifier.postprocessing.data_result_saver import DataResultsSaver
from yolov8_processor.inference.multi_model_inference import MultiModelInference
from yolov8_processor.inference.yolov8_inference import YOLOv8Inference
from yolov8_processor.preprocessing.image_processor import ImageProcessor
from yolov8_processor.postprocessing.result_formatter import ResultFormatter
from flood_classifier.postprocessing.classification_formatter import ClassificationFormatter
from flood_classifier.model.model_loader import ModelLoader
from flood_classifier.inference.fusion_strategy import FusionStrategy
from flood_classifier.inference.image_classifier_2 import EnhancedImageClassifier
from flood_classifier.inference.sensor_classifier import SensorClassifier

# Import your configuration.
import config


def process_message(message_payload):
    """
    Process the incoming MQTT message payload and run the full data processing pipeline.
    Expects the payload to be a JSON string containing "image_data", "sensor_data", and "metadata".
    """
    try:
        # Decode payload (if it comes as bytes) and convert to JSON.
        if isinstance(message_payload, bytes):
            message_payload = message_payload.decode("utf-8")
        message_json = json.loads(message_payload)
        print("\n--- Received MQTT Message ---")
        # Optionally print the formatted JSON:
        # print(json.dumps(message_json, indent=4))
    except Exception as e:
        print("Failed to decode MQTT message:", e)
        return

    # -------------------------------------------
    # Validate and Store Data
    # -------------------------------------------
    validator = DataValidator()
    storage_manager = StorageManager()
    if validator.validate(message_json):
        storage_manager.store(message_json)
        print("Data validated and stored successfully.")
    else:
        print("Data validation failed. Ignoring message.")
        return

    # -------------------------------------------
    # Image Preprocessing & YOLOv8 Inference
    # -------------------------------------------
    image_processor = ImageProcessor()
    try:
        preprocessed_image = image_processor.preprocess(message_json["image_data"])
        print("Image preprocessed for inference.")
    except Exception as e:
        print("Error in image preprocessing:", e)
        return

    try:
        multi_inference = MultiModelInference()
        aggregated_results = multi_inference.run_all_inference(preprocessed_image)
        print("YOLOv8 inference completed.")
    except Exception as e:
        print("Error during YOLOv8 inference:", e)
        aggregated_results = None

    # Format YOLO detection results.
    result_formatter = ResultFormatter()
    detection_results = result_formatter.format_results(aggregated_results) if aggregated_results else []
    print("\n--- YOLOv8 Detection Results ---")
    print(detection_results)

    # -------------------------------------------
    # Flood Classification
    # -------------------------------------------
    classification_mode = config.CLASSIFICATION_MODE  # "combined" or "fused"
    classification_formatter = ClassificationFormatter()
    
    if classification_mode == "combined":
        # Combined classification using both sensor and image data.
        classifier_both = ClassifierBoth(
            baseline_calculator=BaselineCalculator(),
            image_classifier=EnhancedImageClassifier()
        )
        try:
            timestamp = datetime.fromisoformat(
                message_json["metadata"]["timestamp"].replace("Z", "+00:00")
            )
        except Exception as e:
            print("Timestamp conversion error:", e)
            return

        combined_result = classifier_both.classify_flood(
            sensor_data=message_json["sensor_data"],
            detection_data=detection_results,
            image_size=config.IMAGE_SIZE,
            timestamp=timestamp
        )
        print("Combined flood classification result:")
        print(combined_result)
        final_pred = combined_result["final_prediction"]
        final_result = classification_formatter.format_output(final_pred)
    elif classification_mode == "fused":
        # Fuse separate sensor-based and image-based predictions.
        try:
            model_loader = ModelLoader()
            sensor_model = model_loader.load_sensor_model()
            sensor_classifier = SensorClassifier()
            sensor_pred = sensor_classifier.predict(sensor_model, message_json["sensor_data"])
            print("Sensor-based flood prediction (0: No Flood, 1: Some Water, 2: Flooded):", sensor_pred)
        except Exception as e:
            print("Error during sensor classification:", e)
            sensor_pred = None

        image_classifier = EnhancedImageClassifier()
        image_pred = image_classifier.classify_flood(detection_results)
        print("Image-based flood prediction (0: No Flood, 1: Some Water, 2: Flooded):", image_pred)

        fusion = FusionStrategy()
        final_pred = fusion.merge_predictions(sensor_pred, image_pred)
        final_result = classification_formatter.format_output(final_pred)
    else:
        print("Invalid classification mode selected. Exiting processing.")
        return

    # Save results.
    saver = DataResultsSaver()
    saver.save(message_json, final_result)

    # -------------------------------------------
    # Update the Baselines Using Latest Data
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


def main():
    """
    Main method for the Raspberry Pi.
    Initializes the MQTT receiver and directs each incoming message payload
    to the process_message callback for processing.
    """
    # Instantiate your MQTTReceiver with the broker configuration.
    receiver = MQTTReceiver(
        broker_url=config.MQTT_BROKER_URL,    # e.g., "192.168.1.100" or a broker domain
        broker_port=config.MQTT_BROKER_PORT,    # e.g., 1883
        topic=config.MQTT_TOPIC                 # e.g., "your/topic/here"
    )

    # Override the on_message callback to use our full pipeline.
    # This custom callback receives the MQTT message and passes its payload to process_message.
    def custom_on_message(client, userdata, msg):
        print(f"Message received on topic: {msg.topic}")
        process_message(msg.payload)
    receiver.on_message = custom_on_message

    print("Starting MQTT receiver. Waiting for messages...")
    receiver.start()


if __name__ == "__main__":
    main()
