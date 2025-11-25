"""
Main entry point for Ablation 1: Static Medium-YOLO Only (Baseline)
- Single medium model, no FSM, no sensor fusion, no offload
"""

# Override config module BEFORE any other imports
import config_ablation2 as config
import sys
sys.modules['config'] = config

# Now import the rest (they will use config_ablation2)
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
import gc

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
from flood_classifier.utils.queue_monitor import QueueMonitor


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


# Global strategy variable to be initialized at startup
_strategy = None

# Global processor instances to reuse (avoid creating new instances each message)
_image_processor = None
_multi_inference = None
_result_formatter = None


def get_processors():
    """
    Get or create global processor instances for image processing.
    These are reused across all messages to avoid memory overhead.
    """
    global _image_processor, _multi_inference, _result_formatter
    
    if _image_processor is None:
        from yolov8_processor.preprocessing.image_processor import ImageProcessor
        _image_processor = ImageProcessor()
    
    if _multi_inference is None:
        from yolov8_processor.inference.multi_model_inference import MultiModelInference
        _multi_inference = MultiModelInference()
        # Prewarm inference to eliminate cold-start delays
        if getattr(config, "PREWARM_MODELS", True):
            _multi_inference.prewarm_inference()
    
    if _result_formatter is None:
        from yolov8_processor.postprocessing.result_formatter import ResultFormatter
        _result_formatter = ResultFormatter()
    
    return _image_processor, _multi_inference, _result_formatter

def process_message(message_payload, image_name=None, queue_wait_s: float = 0.0, queue_enter_ts: Optional[float] = None, strategy=None):
    """
    Process the incoming MQTT message payload and run the full data processing pipeline.
    Expects the payload to be a JSON string containing "image_data", "sensor_data", and "metadata".
    
    Args:
        message_payload: The MQTT message payload
        image_name: Optional image name
        queue_wait_s: Queue wait time in seconds
        queue_enter_ts: Queue entry timestamp
        strategy: Classification strategy (uses global _strategy if None)
    """
    # Use provided strategy or global one
    active_strategy = strategy or _strategy
    if active_strategy is None:
        raise RuntimeError("Classification strategy not initialized. Call initialize_strategy() first.")
    
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
        # Debug: Frame processing started
        print(f"⏱️  [PI] Frame processing started (JSON parse: {json_parse_duration*1000:.1f}ms)")
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
    preprocessed_image = None
    aggregated_results = None
    classification_mode = config.CLASSIFICATION_MODE  # "yolo_sensor" or "llm_only"
    
    try:
        if classification_mode == "yolo_sensor":
            # Reuse global processor instances (memory efficient)
            image_processor, multi_inference, result_formatter = get_processors()
            
            try:
                preprocess_start = time.perf_counter()
                preprocessed_image = image_processor.preprocess(message_json["image_data"])
                preprocess_time = time.perf_counter() - preprocess_start
                print("Image preprocessed for inference.")
                # DELETE base64 data immediately - we have the decoded image now
                # This frees significant memory (200-500KB per image) early
                if "image_data" in message_json:
                    del message_json["image_data"]
                    gc.collect()  # Force immediate cleanup of base64 string
            except Exception as e:
                print("Error in image preprocessing:", e)
                return

            try:
                aggregated_results = multi_inference.run_all_inference(
                    preprocessed_image,
                    image_name,
                    metadata=message_json.get("metadata"),
                )
                print("YOLOv8 inference completed.")
            except Exception as e:
                print("Error during YOLOv8 inference:", e)
                aggregated_results = None

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
    finally:
        # Explicitly clean up large image objects as early as possible
        # Note: We can't delete preprocessed_image yet if it's needed for classification
        # but we'll clean it up after classification
        pass
    # -------------------------------------------
    # Flood Classification
    # -------------------------------------------
    strategy_map = {
        "yolo_sensor": YoloSensorStrategy,
        "llm_only": LLMOnlyStrategy,
        "fsm": FSMStrategy,
    }
    # Use the pre-initialized strategy (created at startup)
    # Debug: Starting classification
    classification_start = time.perf_counter()
    print(f"🧠 [PI] Starting classification (mode: {classification_mode})...")
    
    final_result = active_strategy.classify(detection_results, message_json)
    
    # Debug: Classification complete
    classification_time = time.perf_counter() - classification_start
    print(f"✅ [PI] Classification complete in {classification_time*1000:.1f}ms")

    if isinstance(final_result, FrameDecision):
        printable_result = decision_to_dict(final_result)
    elif is_dataclass(final_result):
        printable_result = asdict(final_result)
    else:
        printable_result = final_result

    # Save results.
    save_start = time.perf_counter()
    storage_dir = getattr(config, 'RESULTS_STORAGE_DIR', 'storage/data_results')
    video_storage_dir = getattr(config, 'RESULTS_VIDEO_STORAGE_DIR', 'storage/video_results')
    saver = DataResultsSaver(storage_dir=storage_dir, video_storage_dir=video_storage_dir)
    
    # Create result_payload WITHOUT image_data to save memory
    # The saver doesn't need image_data, only sensor_data, metadata, and timing
    result_payload = {
        "sensor_data": message_json.get("sensor_data", {}),
        "metadata": merge_metadata(original_metadata, message_json.get("metadata")),
    }
    # Don't copy image_data - it's not needed and wastes memory
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
    # Explicit Memory Cleanup - Clean up large objects immediately
    # -------------------------------------------
    # Clean up large image objects now that we're done with them
    if preprocessed_image is not None:
        del preprocessed_image
    if aggregated_results is not None:
        del aggregated_results
    # Remove image_data from message_json to free memory
    if "image_data" in message_json:
        del message_json["image_data"]
    # Clean up other large objects
    if detection_results:
        del detection_results
    del result_payload
    
    # Force garbage collection periodically to free memory immediately
    # This helps prevent memory pressure from building up
    gc.collect()

    # -------------------------------------------
    # Update the Baselines Using Latest Data
    # -------------------------------------------
    storage_dir = getattr(config, 'RESULTS_STORAGE_DIR', 'storage/data_results')
    baseline_calculator = BaselineCalculator(
        results_dir=storage_dir,
        baseline_file="storage/sensor_baselines.json",
        tau=12
    )
    baseline_start = time.perf_counter()
    
    # Check if baseline updates are enabled (disabled during evaluation for performance)
    if getattr(config, 'ENABLE_BASELINE_UPDATES', True):
        print("\n🔄 Updating sensor baselines...")
        current_baselines = baseline_calculator.update_baselines()
        baseline_duration = time.perf_counter() - baseline_start
        timing_payload["baseline_update_s"] = baseline_duration
        
        # Debug: Baseline update timing
        print(f"⏱️  [PI] Baseline update took {baseline_duration*1000:.1f}ms")
    else:
        # Just read existing baselines (much faster - no file scanning)
        current_baselines = baseline_calculator.get_baselines()
        baseline_duration = time.perf_counter() - baseline_start
        timing_payload["baseline_lookup_s"] = baseline_duration  # Just lookup, not update
        
        # Debug: Baseline lookup timing
        print(f"⏱️  [PI] Baseline lookup took {baseline_duration*1000:.1f}ms")
    
    # Calculate total pipeline latency including everything
    total_pipeline_latency = time.perf_counter() - start_time
    timing_payload["total_pipeline_latency_s"] = total_pipeline_latency
    
    # Debug: Total frame processing time
    fsm_time = timing_payload.get('fsm_classification_core_s', 0)
    if fsm_time == 0 and 'classification_time' in locals():
        fsm_time = classification_time
    print(f"🎯 [PI] Total frame processing: {total_pipeline_latency*1000:.1f}ms")
    print(f"   Breakdown: json={timing_payload.get('json_parse_s', 0)*1000:.1f}ms, "
          f"validation={timing_payload.get('validation_storage_s', 0)*1000:.1f}ms, "
          f"classification={fsm_time*1000:.1f}ms, "
          f"baseline={baseline_duration*1000:.1f}ms")
    
    if getattr(config, 'ENABLE_BASELINE_UPDATES', True):
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


def initialize_strategy():
    """Initialize the classification strategy at startup to trigger pre-warming."""
    global _strategy
    if _strategy is not None:
        return _strategy
    
    print("\n🚀 Initializing classification strategy...")
    strategy_map = {
        "yolo_sensor": YoloSensorStrategy,
        "llm_only": LLMOnlyStrategy,
        "fsm": FSMStrategy,
    }
    classification_mode = getattr(config, 'CLASSIFICATION_MODE', 'fsm')
    strategy_cls = strategy_map.get(classification_mode)
    if strategy_cls is None:
        raise ValueError(f"Invalid classification mode: {classification_mode}")
    
    _strategy = strategy_cls()
    print("✅ Strategy initialized (models pre-warmed if applicable)\n")
    return _strategy

def main():
    """
    Main method for the Raspberry Pi.
    Initializes the MQTT receiver and directs each incoming message payload
    to the process_message callback for processing.
    """
    print("=" * 60)
    print("ABLATION 2: Vision-Only + FSM + Multi-Model + Offload")
    print("=" * 60)
    
    # Debug: Print config file being used
    config_file = getattr(config, '__file__', 'unknown')
    print(f"📋 [PI] Using config file: {config_file}")
    print(f"📋 [PI] Config settings:")
    print(f"   - CLASSIFICATION_MODE: {getattr(config, 'CLASSIFICATION_MODE', 'unknown')}")
    print(f"   - model_size: {getattr(config, 'model_size', 'unknown')}, model_number: {getattr(config, 'model_number', 'unknown')}")
    print(f"   - DISABLE_SENSOR_FUSION: {getattr(config, 'DISABLE_SENSOR_FUSION', 'unknown')}")
    print(f"   - INFERENCE_ROUTING: {getattr(config, 'INFERENCE_ROUTING', {})}")
    print(f"   - MQTT_BROKER_URL: {getattr(config, 'MQTT_BROKER_URL', 'unknown')}")
    print(f"   - ENABLE_BASELINE_UPDATES: {getattr(config, 'ENABLE_BASELINE_UPDATES', True)}")
    print("=" * 60)
    
    # Initialize strategy at startup (triggers pre-warming for FSM mode)
    initialize_strategy()
    
    # Instantiate your MQTTReceiver with the broker configuration.
    receiver = MQTTReceiver(
        broker_url=config.MQTT_BROKER_URL,
        broker_port=config.MQTT_BROKER_PORT,
        topic=config.MQTT_TOPIC,
    )

    config.IMAGE_MODE = "MQTT_Final"

    latest_messages = LatestPayloadBuffer()
    queue_monitor = QueueMonitor()
    stop_event = threading.Event()

    def custom_on_message(_client, _userdata, msg):
        print(f"Message received on topic: {msg.topic}")
        entry_ts = queue_monitor.enter()
        superseded = latest_messages.offer(msg.topic, msg.payload)
        if superseded is not None:
            print("Superseded older collector payload; keeping newest frame only.")
        
        # Check for queue alerts
        alert = queue_monitor.check_alert_threshold(threshold_depth=5, threshold_wait_s=0.5)
        if alert:
            print(alert)

    receiver.on_message = custom_on_message

    def worker_loop() -> None:
        while not stop_event.is_set():
            item = latest_messages.take()
            if item is None:
                continue
            topic, payload, offered_perf, offered_ts = item
            queue_wait_s = queue_monitor.exit(offered_perf) if offered_perf is not None else 0.0
            
            # Log queue stats periodically
            stats = queue_monitor.get_stats()
            if stats["total_entries"] % 10 == 0:  # Every 10 messages
                print(f"📊 Queue stats: depth={stats['current_depth']:.0f}, "
                      f"avg_wait={stats['avg_wait_s']*1000:.1f}ms, "
                      f"max_wait={stats['max_wait_s']*1000:.1f}ms")
                
                # Memory monitoring - check memory usage periodically
                # Monitor BOTH process and system-wide memory to detect OOM conditions
                try:
                    import psutil
                    import os
                    process = psutil.Process(os.getpid())
                    
                    # Check BOTH process and system memory
                    process_mem_mb = process.memory_info().rss / 1024 / 1024
                    system_mem = psutil.virtual_memory()
                    system_mem_mb = system_mem.total / 1024 / 1024
                    system_mem_available_mb = system_mem.available / 1024 / 1024
                    system_mem_percent = system_mem.percent
                    
                    # Alert if system memory is low (more critical than process memory)
                    # This detects OOM conditions that cause system freezes
                    if system_mem_percent > 85 or system_mem_available_mb < 200:
                        print(f"🚨 CRITICAL SYSTEM MEMORY: {system_mem_percent:.1f}% used, {system_mem_available_mb:.0f}MB available - forcing aggressive cleanup")
                        gc.collect()
                    elif process_mem_mb > 800:  # Lower threshold for Pi (was 1000MB)
                        print(f"⚠️  HIGH PROCESS MEMORY: {process_mem_mb:.0f}MB - forcing garbage collection")
                        gc.collect()
                    elif stats["total_entries"] % 20 == 0:  # More frequent logging (was every 50)
                        print(f"💾 Memory: process={process_mem_mb:.0f}MB, system={system_mem_percent:.1f}% ({system_mem_available_mb:.0f}MB free)")
                except ImportError:
                    # psutil not available, skip memory monitoring
                    pass
                except Exception as e:
                    # Don't fail if memory monitoring fails
                    print(f"Memory monitoring error: {e}")
                    pass
            
            try:
                config.IMAGE_NAME = (
                    "MQTT_Image"
                    + str(Timestamp.now().date())
                    + str(Timestamp.now().time())
                )
            except Exception:
                config.IMAGE_NAME = "MQTT_Image"
            
            # Clean up payload reference after processing
            try:
                process_message(
                    payload,
                    image_name=config.IMAGE_NAME,
                    queue_wait_s=queue_wait_s,
                    queue_enter_ts=offered_ts,
                )
            except Exception as exc:
                print(f"Error processing buffered message: {exc}")
                import traceback
                traceback.print_exc()
            finally:
                # Clean up payload reference
                del payload
                # Garbage collection only when memory is high (not every 5 messages)
                try:
                    import psutil
                    import os
                    process = psutil.Process(os.getpid())
                    process_mem_mb = process.memory_info().rss / 1024 / 1024
                    if process_mem_mb > 600:  # Only when memory is actually high
                        gc.collect()
                except (ImportError, Exception):
                    # If psutil not available or error, skip GC check
                    pass

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

