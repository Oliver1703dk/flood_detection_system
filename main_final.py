from datetime import datetime
import os
import json
import base64
from sqlite3.dbapi2 import Timestamp
import cv2
import numpy as np

# Import your project modules.
from cluster_data_receiver.receiver.mqtt_receiver import MQTTReceiver
from cluster_data_receiver.validation.data_validator import DataValidator
from cluster_data_receiver.storage.storage_manager import StorageManager
from flood_classifier.baselinecalculator.baseline_calculator import BaselineCalculator
from flood_classifier.postprocessing.data_result_saver import DataResultsSaver
from yolov8_processor.inference.multi_model_inference import MultiModelInference
from yolov8_processor.inference.yolov8_inference import YOLOv8Inference
from yolov8_processor.preprocessing.image_processor import ImageProcessor
from yolov8_processor.postprocessing.result_formatter import ResultFormatter
from flood_classifier.classification.strategies import (
    YoloSensorStrategy,
    LLMOnlyStrategy,
)

# Import your configuration.
import config


def process_message(message_payload, image_name=None):
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
        aggregated_results = multi_inference.run_all_inference(preprocessed_image, image_name)
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
    classification_mode = config.CLASSIFICATION_MODE  # "yolo_sensor" or "llm_only"
    strategy_map = {
        "yolo_sensor": YoloSensorStrategy,
        "llm_only": LLMOnlyStrategy,
    }
    strategy_cls = strategy_map.get(classification_mode)
    if strategy_cls is None:
        print("Invalid classification mode selected. Exiting processing.")
        return

    strategy = strategy_cls()
    final_result = strategy.classify(detection_results, message_json)

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

    # Set correct config variables
    config.IMAGE_MODE = "MQTT_Final"  # Ensure IMAGE_MODE is set for processing

    # Override the on_message callback to use our full pipeline.
    # This custom callback receives the MQTT message and passes its payload to process_message.
    def custom_on_message(client, userdata, msg):
        config.IMAGE_NAME = "MQTT_Image" + str(Timestamp.now().date()) + str(Timestamp.now().time())  # Set a default image name for testing
        image_name = config.IMAGE_NAME
        print(f"Message received on topic: {msg.topic}")
        process_message(msg.payload, image_name=image_name)
    receiver.on_message = custom_on_message

    print("Starting MQTT receiver. Waiting for messages...")
    receiver.start()


if __name__ == "__main__":
    main()
