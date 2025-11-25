"""Jetson-side MQTT worker handling heavyweight inference tasks.

The implementation mirrors the Pi logic so that detections and LLM predictions
share the exact schema expected by the downstream FSM on the Raspberry Pi.
"""
from __future__ import annotations

import base64
import json
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import paho.mqtt.client as mqtt

# Ensure imports resolve when the script is launched from inside jetson_worker.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

import config
from flood_classifier.fsm.flood_fsm import FSMParams, ModelTier
from yolov8_processor.classifier.yolov8_final_classifier import YOLOv8FinalClassifier
from yolov8_processor.inference.yolov8_inference import YOLOv8Inference
from yolov8_processor.postprocessing.result_formatter import ResultFormatter
from yolov8_processor.preprocessing.image_processor import ImageProcessor
from jetson_worker.power_monitor import get_power_monitor

BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "localhost")
BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", "1883"))
REQUEST_TOPIC = os.getenv("MQTT_INFERENCE_REQUEST_TOPIC", "inference/request")
RESPONSE_TOPIC = os.getenv("MQTT_INFERENCE_RESPONSE_TOPIC", "inference/response")
HEARTBEAT_TOPIC = os.getenv("MQTT_INFERENCE_HEARTBEAT_TOPIC", "inference/jetson/status")
QOS = int(os.getenv("MQTT_INFERENCE_QOS", "1"))
HEARTBEAT_INTERVAL = float(os.getenv("MQTT_HEARTBEAT_SECONDS", "5"))


def _log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] [JetsonWorker] {message}", flush=True)


_IMAGE_PROCESSOR = ImageProcessor(target_size=getattr(config, "IMAGE_SIZE", (640, 640)))
_RESULT_FORMATTER = ResultFormatter()
_FSM_PARAMS = FSMParams()
_TIER_MODEL_PATHS = {tier.value: path for tier, path in _FSM_PARAMS.tier_paths.items()}
for extra_tier in ("large", "xlarge"):
    _TIER_MODEL_PATHS.setdefault(extra_tier, f"{extra_tier}/best1.pt")
_YOLO_MODEL_ROOT = Path(__file__).resolve().parent.parent / "yolov8_processor" / "model"
_YOLO_MODELS: Dict[str, List[YOLOv8Inference]] = {}
_YOLO_LOCK = threading.Lock()
_YOLO_AGGREGATOR = YOLOv8FinalClassifier()
_LLM_CLASSIFIER: Optional[Any] = None
_LLM_LOCK = threading.Lock()


def _prewarm_yolo_models() -> Dict[str, float]:
    """Pre-load all available YOLO models at startup to avoid cold-start delays.
    
    Also runs dummy inference to fully initialize GPU/CUDA context, TensorRT engines,
    and allocate GPU memory. This eliminates cold-start delays on first real inference.
    
    Returns:
        Dictionary mapping tier to load time in seconds
    """
    _log("🔥 Pre-warming YOLO models...")
    load_times: Dict[str, float] = {}
    
    # Create a dummy image for GPU warmup (matches standard input size from config)
    image_size = getattr(config, "IMAGE_SIZE", (640, 640))
    dummy_image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    _log(f"📐 Created dummy image for warmup: {image_size[0]}x{image_size[1]}")
    
    for tier_value in _TIER_MODEL_PATHS.keys():
        if tier_value in _YOLO_MODELS:
            _log(f"Model {tier_value} already loaded, skipping prewarm")
            continue
        
        try:
            load_start = time.perf_counter()
            model_filenames = _discover_model_filenames(tier_value)
            models = []
            for model_index, filename in enumerate(model_filenames, start=1):
                model_path = _YOLO_MODEL_ROOT / filename
                if not model_path.exists():
                    _log(f"Warning: model file '{model_path}' not found; skipping")
                    continue
                models.append(YOLOv8Inference(str(model_path), identifier=str(model_index)))
            
            if models:
                _YOLO_MODELS[tier_value] = models
                load_time = time.perf_counter() - load_start
                load_times[tier_value] = load_time
                _log(f"✅ Loaded {tier_value} model ({len(models)} model(s)) in {load_time:.3f}s")
                
                # Run dummy inference to fully initialize GPU/CUDA
                warmup_start = time.perf_counter()
                _log(f"🔥 Running GPU warmup inference for {tier_value} model...")
                try:
                    for idx, model in enumerate(models, start=1):
                        # Run a dummy inference to trigger GPU initialization
                        # This initializes CUDA kernels, TensorRT engines, GPU memory allocation, etc.
                        _ = model.model(dummy_image)  # Direct YOLO call - triggers full GPU init
                        _log(f"  ✓ Model {idx}/{len(models)} GPU warmup complete")
                    warmup_time = time.perf_counter() - warmup_start
                    _log(f"✅ GPU warmup complete for {tier_value} in {warmup_time:.3f}s")
                except Exception as warmup_exc:
                    warmup_time = time.perf_counter() - warmup_start
                    _log(f"⚠️ GPU warmup failed for {tier_value} after {warmup_time:.3f}s: {warmup_exc}")
                    _log(f"   (Models loaded but may have cold-start on first real inference)")
            else:
                _log(f"⚠️ No models found for tier {tier_value}")
        except Exception as exc:
            _log(f"⚠️ Failed to pre-warm {tier_value} model: {exc}")
    
    if load_times:
        total_time = sum(load_times.values())
        _log(f"✅ Pre-warmed {len(load_times)} tier(s) in {total_time:.3f}s total")
        _log("🎯 All models ready - GPU fully initialized, no cold-start delays expected")
    else:
        _log("ℹ️ No models to pre-warm")
    
    return load_times


def _discover_model_filenames(tier_value: str) -> List[str]:
    """Return all checkpoint filenames for the requested tier."""

    default_path = _TIER_MODEL_PATHS[tier_value]
    relative_path = Path(default_path)
    tier_subdir = relative_path.parent
    tier_dir = (_YOLO_MODEL_ROOT / tier_subdir).resolve()

    if not tier_dir.exists():
        return [default_path]

    candidates = sorted(tier_dir.glob("best*.pt"))
    if not candidates:
        return [default_path]

    candidate_names = [candidate.name for candidate in candidates]
    if not str(tier_subdir) or str(tier_subdir) == ".":
        return candidate_names

    return [f"{tier_subdir.as_posix()}/{name}" for name in candidate_names]


def run_yolo_inference(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Decode the request image and run YOLOv8 for the requested tier."""

    tier_value = str(payload.get("tier", config.LOCAL_YOLO_TIER)).lower()
    image_b64 = payload.get("image_b64")
    if not image_b64:
        raise ValueError("Missing image data for YOLO inference")

    if tier_value not in _TIER_MODEL_PATHS:
        raise ValueError(f"Unsupported tier '{tier_value}' in request")

    # Debug: Start YOLO inference
    _log(f"🔍 [JETSON] YOLO inference: tier={tier_value}, image_size={len(image_b64)} bytes")

    sensor_data = payload.get("sensor_data") or payload.get("metadata", {}).get("sensor_data")
    sensor_baseline = payload.get("sensor_baseline") or payload.get("metadata", {}).get("sensor_baseline")
    metadata = payload.get("metadata") or {}

    metrics = payload.setdefault("_timing", {})
    preprocess_start = time.perf_counter()
    
    # Debug: Preprocessing
    _log(f"📸 [JETSON] Preprocessing image...")
    
    # Pass timing_dict to image processor to collect detailed timing
    image = _IMAGE_PROCESSOR.preprocess(image_b64, timing_dict=metrics)
    metrics["preprocess_s"] = time.perf_counter() - preprocess_start
    
    # Debug: Preprocessing complete
    _log(f"⏱️  [JETSON] Preprocessing took {metrics['preprocess_s']*1000:.1f}ms")
    
    if image is None:
        raise ValueError("Failed to decode image for YOLO inference")

    inference_start = time.perf_counter()
    # Start power monitoring for YOLO inference
    power_monitor = get_power_monitor()
    job_id = f"yolo_{tier_value}_{int(time.time() * 1000)}"
    power_monitor.start_sampling(job_id)
    
    # Debug: Starting inference
    _log(f"🤖 [JETSON] Starting YOLO inference (tier={tier_value})...")
    
    with _YOLO_LOCK:
        models = _YOLO_MODELS.get(tier_value)
        if not models:
            _log(f"Loading YOLO models for tier '{tier_value}'")
            model_filenames = _discover_model_filenames(tier_value)
            models = []
            for model_index, filename in enumerate(model_filenames, start=1):
                model_path = _YOLO_MODEL_ROOT / filename
                if not model_path.exists():
                    _log(f"Warning: model file '{model_path}' not found; skipping")
                    continue
                models.append(YOLOv8Inference(str(model_path), identifier=str(model_index)))
            if not models:
                raise FileNotFoundError(f"No valid YOLO models found for tier '{tier_value}'")
            _YOLO_MODELS[tier_value] = models
            _log(f"Loaded {len(models)} YOLO model(s) for tier '{tier_value}'")

        # Debug: Model count
        model_count = len(models)
        _log(f"🔄 [JETSON] Running {model_count} model(s)...")

        run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        results_by_model = {}
        for idx, model in enumerate(models, start=1):
            # Debug: Individual model inference
            model_start = time.perf_counter()
            _log(f"  → Model {idx}/{model_count} inference...")
            
            results_by_model[idx] = model.run_inference(
                image,
                run_id=run_id,
                metadata=metadata,
            )
            
            # Debug: Model complete
            model_time = time.perf_counter() - model_start
            _log(f"  ✓ Model {idx} completed in {model_time*1000:.1f}ms")

        # Debug: Aggregation
        _log(f"🔀 [JETSON] Aggregating {model_count} model results...")
        
        results = _YOLO_AGGREGATOR.classify(results_by_model)
        _YOLO_AGGREGATOR.draw_aggregated_bounding_boxes(
            image,
            results,
            run_id=run_id,
            metadata=metadata,
        )
    metrics["inference_s"] = time.perf_counter() - inference_start
    
    # Stop power monitoring and add metrics to timing
    power_metrics = power_monitor.stop_sampling(job_id)
    if power_metrics["energy_j"] is not None:
        metrics["backend_energy_j"] = power_metrics["energy_j"]
        metrics["backend_avg_power_w"] = power_metrics["avg_power_w"]
        metrics["backend_peak_power_w"] = power_metrics["peak_power_w"]
        metrics["backend_cpu_util_%"] = power_metrics["cpu_util_%"]
        metrics["backend_gpu_util_%"] = power_metrics["gpu_util_%"]
    
    format_start = time.perf_counter()
    formatted = _RESULT_FORMATTER.format_results(results)
    metrics["result_format_s"] = time.perf_counter() - format_start

    # Ensure model provenance is carried through for downstream fusion logic.
    for det in formatted:
        model_ids = det.setdefault("model_ids", [])
        if tier_value not in model_ids:
            model_ids.append(tier_value)

    # Debug: Inference complete
    image_name = payload.get("metadata", {}).get("image_name", "unknown")
    _log(f"✅ [JETSON] YOLO inference complete: {metrics['inference_s']*1000:.1f}ms, {len(formatted)} detections")
    _log(f"YOLO inference complete for image '{image_name}'; {len(formatted)} detection(s) produced")
    return {
        "detections": formatted,
        "tier": tier_value,
        "sensor_data": sensor_data,
        "sensor_baseline": sensor_baseline,
    }


from jetson_worker.llm.llm_factory import create_llm_classifier

# Replace the run_llm_inference function:
def run_llm_inference(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Mirror the Pi's LLM image classifier for confirmation requests."""

    image_b64 = payload.get("image_b64")
    if not image_b64:
        raise ValueError("Missing image data for LLM inference")

    try:
        metrics = payload.setdefault("_timing", {})
        decode_start = time.perf_counter()
        image_bytes = base64.b64decode(image_b64)
        decode_duration = time.perf_counter() - decode_start
        metrics["llm_image_decode_s"] = decode_duration
        metrics["preprocess_s"] = metrics.get("preprocess_s", 0.0) + decode_duration
    except Exception as exc:
        raise ValueError("Invalid base64 image data for LLM inference") from exc

    sensor_data = payload.get("sensor_data") or payload.get("metadata", {}).get("sensor_data")
    sensor_baseline = payload.get("sensor_baseline") or payload.get("metadata", {}).get("sensor_baseline")
    sensor_anomalies = payload.get("sensor_anomalies")

    # Start power monitoring for LLM inference
    power_monitor = get_power_monitor()
    job_id = f"llm_{int(time.time() * 1000)}"
    power_monitor.start_sampling(job_id)

    # Check if LLM confirmation is enabled before initializing
    if not getattr(config, "USE_LLM_CONFIRMATION", False):
        raise ValueError("LLM inference requested but USE_LLM_CONFIRMATION is disabled in config")
    
    with _LLM_LOCK:
        global _LLM_CLASSIFIER
        if _LLM_CLASSIFIER is None:
            _log("Initializing LLM image classifier")
            # Use factory to create the appropriate classifier
            _LLM_CLASSIFIER = create_llm_classifier(
                raise_exceptions=True,
            )
        llm_start = time.perf_counter()
        prediction = _LLM_CLASSIFIER.classify_flood(
            image_bytes,
            sensor_data=sensor_data,
            sensor_baseline=sensor_baseline,
            sensor_anomalies=sensor_anomalies,
        )
        metrics["llm_inference_s"] = time.perf_counter() - llm_start

    # Stop power monitoring and add metrics to timing
    power_metrics = power_monitor.stop_sampling(job_id)
    if power_metrics["energy_j"] is not None:
        metrics["llm_energy_j"] = power_metrics["energy_j"]
        metrics["llm_avg_power_w"] = power_metrics["avg_power_w"]
        metrics["llm_peak_power_w"] = power_metrics["peak_power_w"]
        metrics["llm_cpu_util_%"] = power_metrics["cpu_util_%"]
        metrics["llm_gpu_util_%"] = power_metrics["gpu_util_%"]

    return {"prediction": prediction}


class JetsonWorker:
    def __init__(self, preload_llm: bool = True) -> None:
        client_id = f"jetson-worker-{int(time.time())}"
        self.client = mqtt.Client(client_id=client_id)
        self.client.enable_logger()
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        _log(f"Connecting to MQTT broker at {BROKER_HOST}:{BROKER_PORT} as '{client_id}'")
        self.client.connect(BROKER_HOST, BROKER_PORT)
        self._stop = threading.Event()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_announced = False
        # Only preload LLM if both preload_llm is True AND USE_LLM_CONFIRMATION is enabled
        self.preload_llm = preload_llm and getattr(config, "USE_LLM_CONFIRMATION", False)
        self._job_lock = threading.Lock()
        self._job_event = threading.Event()
        self._latest_job: Optional[Dict[str, Any]] = None
        self._job_thread = threading.Thread(target=self._job_loop, name="jetson-job-worker", daemon=True)
        
        # Pre-warm YOLO models at startup if enabled
        if getattr(config, "PREWARM_MODELS", True):
            _prewarm_yolo_models()

    def start(self) -> None:
        _log("Starting Jetson worker threads")
        self._heartbeat_thread.start()
        self._job_thread.start()
        self.client.loop_start()
        self.client.subscribe(REQUEST_TOPIC, qos=QOS)
        _log(f"Subscribed to request topic '{REQUEST_TOPIC}' (QoS {QOS})")

        # Pre-load LLM model after MQTT is ready
        if self.preload_llm:
            _log("Pre-loading LLM model (this may take 15-20 seconds)...")
            self._preload_llm()
            _log("✓ LLM model p½re-loaded and ready for inference requests!")
        
        while not self._stop.is_set():
            time.sleep(0.1)

    def _preload_llm(self) -> None:
        """Pre-load the LLM model during startup."""
        global _LLM_CLASSIFIER
        with _LLM_LOCK:
            if _LLM_CLASSIFIER is None:
                _LLM_CLASSIFIER = create_llm_classifier(
                    raise_exceptions=True,
                )
                # Trigger model initialization
                _LLM_CLASSIFIER._initialize_model()

    def stop(self) -> None:
        _log("Stopping Jetson worker")
        self._stop.set()
        self._job_event.set()
        self.client.loop_stop()
        self.client.disconnect()
        self._job_thread.join(timeout=5.0)

    def _on_connect(self, client, _userdata, _flags, rc):
        if rc == 0:
            _log("Connected to MQTT broker successfully")
            client.subscribe(REQUEST_TOPIC, qos=QOS)
        else:
            _log(f"MQTT connection failed with return code {rc}")

    def _on_message(self, client, _userdata, message):
        deserialize_start = time.perf_counter()
        try:
            payload = json.loads(message.payload.decode("utf-8"))
        except json.JSONDecodeError:
            _log("Received malformed JSON payload; ignoring message")
            return

        job_id = payload.get("id")
        task = payload.get("task")

        _log(f"Received task '{task}' (job id: {job_id}); queueing for worker thread")
        metrics = payload.setdefault("_timing", {})
        metrics["mqtt_request_json_deserialize_s"] = time.perf_counter() - deserialize_start
        receive_ts = time.time()
        receive_perf = time.perf_counter()
        metrics["mqtt_receive_ts"] = receive_ts
        metrics["mqtt_receive_perf"] = receive_perf
        sent_ts = payload.get("ts") or metrics.get("sent_ts")
        if isinstance(sent_ts, (int, float)):
            metrics["sent_ts"] = sent_ts
            metrics["network_pi_to_jetson_s"] = max(0.0, receive_ts - sent_ts)
        metrics["received_ts"] = receive_ts
        metrics["received_perf"] = receive_perf
        superseded = self._enqueue_job(payload)
        if superseded is not None:
            self._publish_superseded(superseded)

    def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            self.client.publish(HEARTBEAT_TOPIC, json.dumps({"ts": time.time()}), qos=0, retain=True)
            if not self._heartbeat_announced:
                _log(f"Heartbeat thread active; publishing status beacons to '{HEARTBEAT_TOPIC}' every {HEARTBEAT_INTERVAL}s")
                self._heartbeat_announced = True
            time.sleep(HEARTBEAT_INTERVAL)

    def _enqueue_job(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        with self._job_lock:
            metrics = payload.setdefault("_timing", {})
            metrics["enqueue_ts"] = time.time()
            metrics["enqueue_perf"] = time.perf_counter()
            superseded = self._latest_job
            self._latest_job = payload
            self._job_event.set()
            return superseded

    def _job_loop(self) -> None:
        while True:
            if self._stop.is_set() and not self._job_event.is_set():
                break
            if not self._job_event.wait(timeout=0.5):
                continue
            with self._job_lock:
                job = self._latest_job
                self._latest_job = None
                self._job_event.clear()
            if job is None:
                if self._stop.is_set():
                    break
                continue
            metrics = job.setdefault("_timing", {})
            enqueue_perf = metrics.get("enqueue_perf") or metrics.get("received_perf")
            if enqueue_perf is not None:
                metrics["queue_wait_s"] = max(0.0, time.perf_counter() - enqueue_perf)
            self._process_payload(job)

    def _process_payload(self, payload: Dict[str, Any]) -> None:
        job_id = payload.get("id")
        task = payload.get("task")

        # Debug: Network latency
        receive_time = time.time()
        sent_ts = payload.get("ts") or payload.get("_timing", {}).get("sent_ts")
        if sent_ts:
            network_latency = receive_time - sent_ts
            _log(f"📥 [JETSON] Received {task} job {job_id[:8] if job_id else 'unknown'}... (network: {network_latency*1000:.1f}ms)")
        else:
            _log(f"📥 [JETSON] Received {task} job {job_id[:8] if job_id else 'unknown'}... (network: N/A)")

        _log(f"Processing task '{task}' (job id: {job_id})")
        metrics = payload.setdefault("_timing", {})
        
        # Time JSON deserialization (payload was already deserialized, but we can time it if needed)
        # Note: The payload is already deserialized by MQTT client, so we skip this here
        # but we could add it if we had access to raw payload
        
        metrics["process_start_ts"] = time.time()
        process_start_perf = time.perf_counter()
        
        # Debug: Start processing
        _log(f"⏱️  [JETSON] Starting {task} processing...")
        
        response: Dict[str, Any] = {"id": job_id, "error": None}
        try:
            if task == "yolo":
                response.update(run_yolo_inference(payload))
            elif task == "llm":
                response.update(run_llm_inference(payload))
            else:
                response["error"] = f"Unknown task: {task}"
                _log(f"Unknown task type '{task}' received")
        except Exception as exc:  # pragma: no cover - hardware specific failures
            response["error"] = str(exc)
            _log(f"Task '{task}' failed: {exc}")

        metrics["compute_s"] = time.perf_counter() - process_start_perf
        metrics["process_end_ts"] = time.time()
        total_runtime = metrics.get("compute_s", 0.0)
        if "queue_wait_s" in metrics:
            total_runtime += metrics["queue_wait_s"]
        metrics["total_runtime_s"] = total_runtime
        
        # Debug: Completion
        _log(f"✅ [JETSON] Completed {task} in {metrics['compute_s']*1000:.1f}ms (total: {total_runtime*1000:.1f}ms)")
        total_runtime = metrics.get("compute_s", 0.0)
        if "queue_wait_s" in metrics:
            total_runtime += metrics["queue_wait_s"]
        metrics["total_runtime_s"] = total_runtime

        export_fields = {
            "queue_wait_s": metrics.get("queue_wait_s"),
            "preprocess_s": metrics.get("preprocess_s"),
            "inference_s": metrics.get("inference_s"),
            "result_format_s": metrics.get("result_format_s"),
            "llm_inference_s": metrics.get("llm_inference_s"),
            "compute_s": metrics.get("compute_s"),
            "total_runtime_s": metrics.get("total_runtime_s"),
            "mqtt_request_json_deserialize_s": metrics.get("mqtt_request_json_deserialize_s"),
            "mqtt_response_json_serialize_s": metrics.get("mqtt_response_json_serialize_s"),
            "mqtt_response_publish_s": metrics.get("mqtt_response_publish_s"),
            "mqtt_response_wait_publish_s": metrics.get("mqtt_response_wait_publish_s"),
            "mqtt_receive_ts": metrics.get("mqtt_receive_ts"),
            "mqtt_response_ts": metrics.get("mqtt_response_ts"),
            "network_pi_to_jetson_s": metrics.get("network_pi_to_jetson_s"),
            "sent_ts": metrics.get("sent_ts"),
            "received_ts": metrics.get("received_ts"),
            "process_start_ts": metrics.get("process_start_ts"),
            "process_end_ts": metrics.get("process_end_ts"),
            # Power monitoring metrics
            "backend_energy_j": metrics.get("backend_energy_j"),
            "backend_avg_power_w": metrics.get("backend_avg_power_w"),
            "backend_peak_power_w": metrics.get("backend_peak_power_w"),
            "backend_cpu_util_%": metrics.get("backend_cpu_util_%"),
            "backend_gpu_util_%": metrics.get("backend_gpu_util_%"),
            "llm_energy_j": metrics.get("llm_energy_j"),
            "llm_avg_power_w": metrics.get("llm_avg_power_w"),
            "llm_peak_power_w": metrics.get("llm_peak_power_w"),
            "llm_cpu_util_%": metrics.get("llm_cpu_util_%"),
            "llm_gpu_util_%": metrics.get("llm_gpu_util_%"),
        }
        response["timing"] = {k: v for k, v in export_fields.items() if v is not None}
        for transient_key in ("received_perf", "enqueue_perf"):
            metrics.pop(transient_key, None)

        for transient_key in ("mqtt_receive_perf",):
            metrics.pop(transient_key, None)

        response_topic = f"{RESPONSE_TOPIC.rstrip('/')}/{job_id}" if job_id else RESPONSE_TOPIC

        # Serialize response and publish while capturing timings
        json_start = time.perf_counter()
        response_payload = json.dumps(response).encode("utf-8")
        metrics["mqtt_response_json_serialize_s"] = time.perf_counter() - json_start

        publish_start = time.perf_counter()
        publish_result = self.client.publish(response_topic, response_payload, qos=QOS)
        metrics["mqtt_response_publish_s"] = time.perf_counter() - publish_start

        wait_publish_start = time.perf_counter()
        try:
            publish_result.wait_for_publish()
        except Exception:
            pass
        metrics["mqtt_response_wait_publish_s"] = time.perf_counter() - wait_publish_start
        metrics["mqtt_response_ts"] = time.time()

        _log(f"Published response for job id {job_id} to '{response_topic}'")

    def _publish_superseded(self, payload: Dict[str, Any]) -> None:
        job_id = payload.get("id")
        if not job_id:
            return
        response_topic = f"{RESPONSE_TOPIC.rstrip('/')}/{job_id}"
        response = {"id": job_id, "error": "superseded by newer request"}
        self.client.publish(response_topic, json.dumps(response), qos=QOS)
        _log(f"Superseded job {job_id}; notified publisher")


def main() -> None:
    _log("Launching Jetson inference worker")
    
    # Debug: Print config file being used
    config_file = getattr(config, '__file__', 'unknown')
    _log(f"📋 [JETSON] Using config file: {config_file}")
    _log(f"📋 [JETSON] Config settings:")
    _log(f"   - LOCAL_YOLO_TIER: {getattr(config, 'LOCAL_YOLO_TIER', 'unknown')}")
    _log(f"   - USE_LLM_CONFIRMATION: {getattr(config, 'USE_LLM_CONFIRMATION', False)}")
    _log(f"   - MQTT_BROKER_URL: {getattr(config, 'MQTT_BROKER_URL', 'unknown')}")
    
    # Only preload LLM if USE_LLM_CONFIRMATION is enabled
    preload_llm = getattr(config, "USE_LLM_CONFIRMATION", False)
    worker = JetsonWorker(preload_llm=preload_llm)

    def handle_signal(_sig, _frame):
        _log("Received shutdown signal; stopping worker")
        worker.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    worker.start()


if __name__ == "__main__":
    main()
