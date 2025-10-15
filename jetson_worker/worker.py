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
from pathlib import Path
from typing import Any, Dict, List, Optional

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

    sensor_data = payload.get("sensor_data") or payload.get("metadata", {}).get("sensor_data")
    sensor_baseline = payload.get("sensor_baseline") or payload.get("metadata", {}).get("sensor_baseline")

    image = _IMAGE_PROCESSOR.preprocess(image_b64)
    if image is None:
        raise ValueError("Failed to decode image for YOLO inference")

    with _YOLO_LOCK:
        models = _YOLO_MODELS.get(tier_value)
        if not models:
            _log(f"Loading YOLO models for tier '{tier_value}'")
            model_filenames = _discover_model_filenames(tier_value)
            models = []
            for filename in model_filenames:
                model_path = _YOLO_MODEL_ROOT / filename
                if not model_path.exists():
                    _log(f"Warning: model file '{model_path}' not found; skipping")
                    continue
                models.append(YOLOv8Inference(str(model_path)))
            if not models:
                raise FileNotFoundError(f"No valid YOLO models found for tier '{tier_value}'")
            _YOLO_MODELS[tier_value] = models
            _log(f"Loaded {len(models)} YOLO model(s) for tier '{tier_value}'")

        results_by_model = {}
        for idx, model in enumerate(models):
            results_by_model[idx] = model.run_inference(image)

        if len(results_by_model) > 1:
            results = _YOLO_AGGREGATOR.classify(results_by_model)
        else:
            results = next(iter(results_by_model.values()))

    formatted = _RESULT_FORMATTER.format_results(results)

    # Ensure model provenance is carried through for downstream fusion logic.
    for det in formatted:
        model_ids = det.setdefault("model_ids", [])
        if tier_value not in model_ids:
            model_ids.append(tier_value)

    image_name = payload.get("metadata", {}).get("image_name", "unknown")
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
        image_bytes = base64.b64decode(image_b64)
    except Exception as exc:
        raise ValueError("Invalid base64 image data for LLM inference") from exc

    sensor_data = payload.get("sensor_data") or payload.get("metadata", {}).get("sensor_data")
    sensor_baseline = payload.get("sensor_baseline") or payload.get("metadata", {}).get("sensor_baseline")
    sensor_anomalies = payload.get("sensor_anomalies")

    with _LLM_LOCK:
        global _LLM_CLASSIFIER
        if _LLM_CLASSIFIER is None:
            _log("Initializing LLM image classifier")
            # Use factory to create the appropriate classifier
            _LLM_CLASSIFIER = create_llm_classifier(
                raise_exceptions=False,
            )

        prediction = _LLM_CLASSIFIER.classify_flood(
            image_bytes,
            sensor_data=sensor_data,
            sensor_baseline=sensor_baseline,
            sensor_anomalies=sensor_anomalies,
        )

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
        self.preload_llm = preload_llm
        self._job_lock = threading.Lock()
        self._job_event = threading.Event()
        self._latest_job: Optional[Dict[str, Any]] = None
        self._job_thread = threading.Thread(target=self._job_loop, name="jetson-job-worker", daemon=True)

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
            _log("✓ LLM model pre-loaded and ready for inference requests!")
        
        while not self._stop.is_set():
            time.sleep(0.1)

    def _preload_llm(self) -> None:
        """Pre-load the LLM model during startup."""
        global _LLM_CLASSIFIER
        with _LLM_LOCK:
            if _LLM_CLASSIFIER is None:
                _LLM_CLASSIFIER = create_llm_classifier(
                    raise_exceptions=False,
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
        try:
            payload = json.loads(message.payload.decode("utf-8"))
        except json.JSONDecodeError:
            _log("Received malformed JSON payload; ignoring message")
            return

        job_id = payload.get("id")
        task = payload.get("task")

        _log(f"Received task '{task}' (job id: {job_id}); queueing for worker thread")
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
            self._process_payload(job)

    def _process_payload(self, payload: Dict[str, Any]) -> None:
        job_id = payload.get("id")
        task = payload.get("task")

        _log(f"Processing task '{task}' (job id: {job_id})")
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

        response_topic = f"{RESPONSE_TOPIC.rstrip('/')}/{job_id}" if job_id else RESPONSE_TOPIC
        self.client.publish(response_topic, json.dumps(response), qos=QOS)
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
    worker = JetsonWorker(preload_llm=True)  # Set to False to disable pre-loading

    def handle_signal(_sig, _frame):
        _log("Received shutdown signal; stopping worker")
        worker.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    worker.start()


if __name__ == "__main__":
    main()
