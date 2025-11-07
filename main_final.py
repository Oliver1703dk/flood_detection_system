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
        self._latest: Optional[Tuple[str, bytes, float, float]] = None
        self._available = threading.Event()

    def offer(self, topic: str, payload: bytes) -> Optional[Tuple[str, bytes, float, float]]:
        with self._lock:
            superseded = self._latest
            now_perf = time.perf_counter()
            now_wall = time.time()
            self._latest = (topic, payload, now_perf, now_wall)
            self._available.set()
            return superseded

    def take(self, timeout: float = 0.5) -> Optional[Tuple[str, bytes, float, float]]:
        if not self._available.wait(timeout=timeout):
            return None
        with self._lock:
            item = self._latest
            self._latest = None
            self._available.clear()
            return item


def process_message(message_payload, image_name=None, queue_wait_s: float = 0.0, queue_enter_ts: Optional[float] = None):
    """
    Process the incoming MQTT message payload and run the full data processing pipeline.
    Expects the payload to be a JSON string containing "image_data", "sensor_data", and "metadata".
    """
    process_start_ts = time.time()
    start_time = time.perf_counter()
    timing_payload = {}
    try:
        # Time JSON parsing
        json_parse_start = time.perf_counter()
        # Decode payload (if it comes as bytes) and convert to JSON.
        if isinstance(message_payload, bytes):
            message_payload = message_payload.decode("utf-8")
        message_json = json.loads(message_payload)
        json_parse_duration = time.perf_counter() - json_parse_start
        timing_payload["json_parse_s"] = json_parse_duration
        
        original_metadata = deepcopy(message_json.get("metadata", {}) or {})
        metadata = message_json.setdefault("metadata", {})
        print("\n--- Received MQTT Message ---")
        # Optionally print the formatted JSON:
        # print(json.dumps(message_json, indent=4))
    except Exception as e:
        print("Failed to decode MQTT message:", e)
        return

    if queue_enter_ts is not None:
        timing_payload["processor_queue_enter_ts"] = queue_enter_ts
        metadata["processor_queue_enter_ts"] = queue_enter_ts

    metadata["processor_receive_ts"] = process_start_ts
    if queue_enter_ts is not None:
        metadata["processor_queue_exit_ts"] = queue_enter_ts + queue_wait_s

    def _to_timestamp(value: Optional[object]) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                try:
                    return datetime.fromisoformat(value).timestamp()
                except ValueError:
                    return None
        return None

    collector_capture_ts_val = _to_timestamp(metadata.get("collector_capture_ts"))
    collector_publish_ts_val = _to_timestamp(metadata.get("collector_publish_ts"))

    if collector_capture_ts_val is not None:
        timing_payload["collector_capture_ts"] = collector_capture_ts_val
        metadata["collector_capture_ts"] = collector_capture_ts_val
        timing_payload["collector_capture_to_processor_receive_s"] = process_start_ts - collector_capture_ts_val
        if queue_enter_ts is not None:
            timing_payload["collector_capture_to_queue_enter_s"] = queue_enter_ts - collector_capture_ts_val

    if collector_publish_ts_val is not None:
        timing_payload["collector_publish_ts"] = collector_publish_ts_val
        metadata["collector_publish_ts"] = collector_publish_ts_val
        timing_payload["collector_publish_to_processor_receive_s"] = process_start_ts - collector_publish_ts_val
        if queue_enter_ts is not None:
            timing_payload["collector_publish_to_queue_enter_s"] = queue_enter_ts - collector_publish_ts_val

    # -------------------------------------------
    # Validate and Store Data
    # -------------------------------------------
    validation_start = time.perf_counter()
    validator = DataValidator()
    storage_manager = StorageManager()
    validation_duration = None
    if validator.validate(message_json):
        storage_manager.store(message_json)
        validation_duration = time.perf_counter() - validation_start
        print("Data validated and stored successfully.")
    else:
        print("Data validation failed. Ignoring message.")
        return

    # -------------------------------------------
    # Image Preprocessing & YOLOv8 Inference (only for YOLO+sensor mode)
    # -------------------------------------------
    detection_results = []
    preprocess_time = None
    format_time = None
    classification_mode = config.CLASSIFICATION_MODE  # "yolo_sensor" or "llm_only"
    if classification_mode == "yolo_sensor":
        from yolov8_processor.preprocessing.image_processor import ImageProcessor
        from yolov8_processor.inference.multi_model_inference import MultiModelInference
        from yolov8_processor.postprocessing.result_formatter import ResultFormatter

        image_processor = ImageProcessor()
        try:
            preprocess_start = time.perf_counter()
            preprocessed_image = image_processor.preprocess(message_json["image_data"])
            preprocess_time = time.perf_counter() - preprocess_start
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
        if aggregated_results:
            format_start = time.perf_counter()
            detection_results = result_formatter.format_results(aggregated_results)
            format_time = time.perf_counter() - format_start
        else:
            detection_results = []
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
    save_start = time.perf_counter()
    saver = DataResultsSaver()
    result_payload = dict(message_json)
    result_payload["metadata"] = merge_metadata(original_metadata, message_json.get("metadata"))
    pipeline_latency = time.perf_counter() - start_time
    result_timing = result_payload.setdefault("timing", {})
    # Merge timing_payload (which includes json_parse_s) into result_timing
    result_timing.update(timing_payload)
    result_timing["process_start_ts"] = process_start_ts
    result_timing["pipeline_latency_s"] = pipeline_latency
    result_timing["queue_wait_s"] = queue_wait_s
    if queue_enter_ts is not None:
        queue_exit_ts = queue_enter_ts + queue_wait_s
        result_timing["processor_queue_enter_ts"] = queue_enter_ts
        result_timing["processor_queue_exit_ts"] = queue_exit_ts
    timing_payload = result_timing
    if validation_duration is not None:
        timing_payload["validation_storage_s"] = validation_duration
    if preprocess_time is not None:
        timing_payload["preprocess_s"] = preprocess_time
    if format_time is not None:
        timing_payload["result_format_s"] = format_time
    # Note: classification_s is provided by FSM timing for FSM mode
    if isinstance(final_result, FrameDecision):
        for key, value in final_result.timing.items():
            if isinstance(value, (int, float)):
                # FSM timing fields already have "fsm_" prefix, so just use the key as-is
                timing_payload[key] = value
        backend_info = final_result.backend_info or {}
        if isinstance(backend_info, dict):
            latency_val = backend_info.get("latency_s")
            if isinstance(latency_val, (int, float)):
                timing_payload["backend_latency_s"] = latency_val
            backend_timing = backend_info.get("timing")
            if isinstance(backend_timing, dict):
                for key, value in backend_timing.items():
                    if isinstance(value, (int, float)):
                        timing_payload[f"backend_{key}"] = value
            # Create alias for YOLO energy from backend timing (for local Pi inference)
            if "backend_yolo_energy_j" in timing_payload and "yolo_energy_j" not in timing_payload:
                timing_payload["yolo_energy_j"] = timing_payload["backend_yolo_energy_j"]
            if "backend_yolo_power_w" in timing_payload and "yolo_power_w" not in timing_payload:
                timing_payload["yolo_power_w"] = timing_payload["backend_yolo_power_w"]
            if "backend_yolo_cpu_util_%" in timing_payload and "yolo_cpu_util_%" not in timing_payload:
                timing_payload["yolo_cpu_util_%"] = timing_payload["backend_yolo_cpu_util_%"]
            sent_at = backend_info.get("sent_at")
            received_at = backend_info.get("received_at")
            if isinstance(sent_at, (int, float)) and isinstance(received_at, (int, float)):
                timing_payload["backend_roundtrip_s"] = received_at - sent_at
        llm_backend = final_result.llm_backend_info or {}
        if isinstance(llm_backend, dict):
            llm_latency = llm_backend.get("latency_s")
            if isinstance(llm_latency, (int, float)):
                timing_payload["llm_latency_s"] = llm_latency
            llm_timing = llm_backend.get("timing")
            if isinstance(llm_timing, dict):
                for key, value in llm_timing.items():
                    if isinstance(value, (int, float)):
                        timing_payload[f"llm_backend_{key}"] = value
            llm_sent = llm_backend.get("sent_at")
            llm_received = llm_backend.get("received_at")
            if isinstance(llm_sent, (int, float)) and isinstance(llm_received, (int, float)):
                timing_payload["llm_roundtrip_s"] = llm_received - llm_sent
    
    # Aggregate energy metrics and compute total
    total_energy_j = 0.0
    if isinstance(final_result, FrameDecision):
        # FSM energy
        fsm_energy = timing_payload.get("fsm_energy_j")
        if isinstance(fsm_energy, (int, float)):
            total_energy_j += float(fsm_energy)
        
        # YOLO energy (from local Pi inference)
        yolo_energy = timing_payload.get("yolo_energy_j")
        if isinstance(yolo_energy, (int, float)):
            total_energy_j += float(yolo_energy)
        
        # Backend (Jetson) energy - only if backend was remote
        backend = final_result.backend if hasattr(final_result, 'backend') else backend_info.get("backend", "local")
        if backend == "remote":
            backend_energy = timing_payload.get("backend_energy_j")
            if isinstance(backend_energy, (int, float)):
                total_energy_j += float(backend_energy)
        
        # LLM backend (Jetson) energy - only if LLM was used and it was remote
        if final_result.llm_used:
            llm_backend_energy = timing_payload.get("llm_backend_energy_j")
            if isinstance(llm_backend_energy, (int, float)):
                total_energy_j += float(llm_backend_energy)
        
        # Store total energy
        if total_energy_j > 0:
            timing_payload["total_energy_j"] = total_energy_j
    
    print(f"Pipeline latency: {pipeline_latency:.3f}s (queue wait {queue_wait_s:.3f}s)")
    saved_path = saver.save(result_payload, printable_result)
    save_duration = time.perf_counter() - save_start
    timing_payload["result_save_s"] = save_duration

    # -------------------------------------------
    # Update the Baselines Using Latest Data
    # -------------------------------------------
    baseline_calculator = BaselineCalculator(
        results_dir="storage/data_results",
        baseline_file="storage/sensor_baselines.json",
        tau=12
    )
    print("\n🔄 Updating sensor baselines...")
    baseline_start = time.perf_counter()
    current_baselines = baseline_calculator.update_baselines()
    baseline_duration = time.perf_counter() - baseline_start
    timing_payload["baseline_update_s"] = baseline_duration
    
    # Calculate total pipeline latency including everything
    total_pipeline_latency = time.perf_counter() - start_time
    timing_payload["total_pipeline_latency_s"] = total_pipeline_latency
    
    if current_baselines:
        print("✅ Updated Baselines:", current_baselines)
    else:
        print("❌ Baseline update failed (no stable period found).")
    if saved_path:
        try:
            with open(saved_path, "r+", encoding="utf-8") as f:
                saved_data = json.load(f)
                saved_timing = saved_data.setdefault("timing", {})
                saved_timing["baseline_update_s"] = baseline_duration
                if validation_duration is not None:
                    saved_timing["validation_storage_s"] = validation_duration
                saved_timing["result_save_s"] = save_duration
                saved_timing["total_pipeline_latency_s"] = total_pipeline_latency
                f.seek(0)
                json.dump(saved_data, f, indent=4)
                f.truncate()
        except Exception as exc:
            print(f"Warning: failed to update timing in {saved_path}: {exc}")
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
            topic, payload, offered_perf, offered_ts = item
            queue_wait_s = time.perf_counter() - offered_perf if offered_perf is not None else 0.0
            try:
                config.IMAGE_NAME = (
                    "MQTT_Image"
                    + str(Timestamp.now().date())
                    + str(Timestamp.now().time())
                )
            except Exception:
                config.IMAGE_NAME = "MQTT_Image"
            try:
                process_message(
                    payload,
                    image_name=config.IMAGE_NAME,
                    queue_wait_s=queue_wait_s,
                    queue_enter_ts=offered_ts,
                )
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
