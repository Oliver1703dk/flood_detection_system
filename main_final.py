from datetime import datetime
from dataclasses import asdict, is_dataclass
from copy import deepcopy
import os
import json
import base64
import time
from sqlite3.dbapi2 import Timestamp
import threading
from typing import Optional, Tuple
import cv2
import numpy as np

# Import your project modules.
from cluster_data_receiver.receiver.mqtt_receiver import MQTTReceiver
from cluster_data_receiver.validation.data_validator import DataValidator
from cluster_data_receiver.storage.storage_manager import StorageManager
from flood_classifier.baselinecalculator.baseline_calculator import BaselineCalculator
from flood_classifier.postprocessing.data_result_saver import DataResultsSaver
from flood_classifier.utils.metadata_utils import merge_metadata
from flood_classifier.classification.strategies import (
    YoloSensorStrategy,
    LLMOnlyStrategy,
    FSMStrategy,
)
from flood_classifier.fsm.flood_fsm import FrameDecision, decision_to_dict

# Import your configuration.
import config


class LatestPayloadBuffer:
    """Keep only the most recent MQTT payload for processing."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: Optional[Tuple[str, bytes]] = None
        self._available = threading.Event()

    def offer(self, topic: str, payload: bytes) -> Optional[Tuple[str, bytes]]:
        with self._lock:
            superseded = self._latest
            self._latest = (topic, payload)
            self._available.set()
            return superseded

    def take(self, timeout: float = 0.5) -> Optional[Tuple[str, bytes]]:
        if not self._available.wait(timeout=timeout):
            return None
        with self._lock:
            item = self._latest
            self._latest = None
            self._available.clear()
            return item


def process_message(message_payload, image_name=None):
    """
    Process the incoming MQTT message payload and run the full data processing pipeline.
    Expects the payload to be a JSON string containing "image_data", "sensor_data", and "metadata".
    """
    start_time = time.perf_counter()
    try:
        # Decode payload (if it comes as bytes) and convert to JSON.
        if isinstance(message_payload, bytes):
            message_payload = message_payload.decode("utf-8")
        message_json = json.loads(message_payload)
        original_metadata = deepcopy(message_json.get("metadata", {}) or {})
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
    # Image Preprocessing & YOLOv8 Inference (only for YOLO+sensor mode)
    # -------------------------------------------
    detection_results = []
    classification_mode = config.CLASSIFICATION_MODE  # "yolo_sensor" or "llm_only"
    if classification_mode == "yolo_sensor":
        from yolov8_processor.preprocessing.image_processor import ImageProcessor
        from yolov8_processor.inference.multi_model_inference import MultiModelInference
        from yolov8_processor.postprocessing.result_formatter import ResultFormatter

        image_processor = ImageProcessor()
        try:
            preprocessed_image = image_processor.preprocess(message_json["image_data"])
            print("Image preprocessed for inference.")
        except Exception as e:
            print("Error in image preprocessing:", e)
            return

        try:
            multi_inference = MultiModelInference()
            aggregated_results = multi_inference.run_all_inference(
                preprocessed_image,
                image_name,
                metadata=message_json.get("metadata"),
            )
            print("YOLOv8 inference completed.")
        except Exception as e:
            print("Error during YOLOv8 inference:", e)
            aggregated_results = None

        result_formatter = ResultFormatter()
        detection_results = (
            result_formatter.format_results(aggregated_results)
            if aggregated_results
            else []
        )
        print("\n--- YOLOv8 Detection Results ---")
        print(detection_results)
    else:
        print("Skipping YOLOv8 inference (LLM-only / FSM mode).")
    # -------------------------------------------
    # Flood Classification
    # -------------------------------------------
    strategy_map = {
        "yolo_sensor": YoloSensorStrategy,
        "llm_only": LLMOnlyStrategy,
        "fsm": FSMStrategy,
    }
    strategy_cls = strategy_map.get(classification_mode)
    if strategy_cls is None:
        print("Invalid classification mode selected. Exiting processing.")
        return

    strategy = strategy_cls()
    final_result = strategy.classify(detection_results, message_json)

    if isinstance(final_result, FrameDecision):
        printable_result = decision_to_dict(final_result)
    elif is_dataclass(final_result):
        printable_result = asdict(final_result)
    else:
        printable_result = final_result

    # Save results.
    saver = DataResultsSaver()
    result_payload = dict(message_json)
    result_payload["metadata"] = merge_metadata(original_metadata, message_json.get("metadata"))
    pipeline_latency = time.perf_counter() - start_time
    result_payload.setdefault("timing", {})["pipeline_latency_s"] = pipeline_latency
    print(f"Pipeline latency: {pipeline_latency:.3f}s")
    saver.save(result_payload, printable_result)

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
    print(printable_result)


def main():
    """
    Main method for the Raspberry Pi.
    Initializes the MQTT receiver and directs each incoming message payload
    to the process_message callback for processing.
    """
    # Instantiate your MQTTReceiver with the broker configuration.
    receiver = MQTTReceiver(
        broker_url=config.MQTT_BROKER_URL,
        broker_port=config.MQTT_BROKER_PORT,
        topic=config.MQTT_TOPIC,
    )

    config.IMAGE_MODE = "MQTT_Final"

    latest_messages = LatestPayloadBuffer()
    stop_event = threading.Event()

    def custom_on_message(_client, _userdata, msg):
        print(f"Message received on topic: {msg.topic}")
        superseded = latest_messages.offer(msg.topic, msg.payload)
        if superseded is not None:
            print("Superseded older collector payload; keeping newest frame only.")

    receiver.on_message = custom_on_message

    def worker_loop() -> None:
        while not stop_event.is_set():
            item = latest_messages.take()
            if item is None:
                continue
            _, payload = item
            try:
                config.IMAGE_NAME = (
                    "MQTT_Image"
                    + str(Timestamp.now().date())
                    + str(Timestamp.now().time())
                )
            except Exception:
                config.IMAGE_NAME = "MQTT_Image"
            try:
                process_message(payload, image_name=config.IMAGE_NAME)
            except Exception as exc:
                print(f"Error processing buffered message: {exc}")

    worker_thread = threading.Thread(target=worker_loop, name="processor-worker", daemon=True)
    worker_thread.start()

    print("Starting MQTT receiver. Waiting for messages...")
    try:
        receiver.start()
    except KeyboardInterrupt:
        print("Stopping MQTT receiver (KeyboardInterrupt)")
    finally:
        stop_event.set()
        worker_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
