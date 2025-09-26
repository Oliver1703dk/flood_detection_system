# Flood Detection System Overview

This document summarizes the major components and data flow in the distributed
flood detection setup. The architecture spans two Raspberry Pi devices and an
NVIDIA Jetson, coordinated via MQTT messaging.

## 1. Gathering Pi (Sensor & Image Capture)
- Executes the edge data processor (see `main.py`) which reloads `.env` secrets, reads feature flags from `config.py`, and instantiates handlers for the camera, Netatmo sensors (or simulators), metadata enrichment, and optional MQTT transport.
- The Netatmo integration captures barometric pressure, temperature, humidity, CO₂ concentration, and rainfall metrics; simulation mode synthesizes plausible values within configured bounds so downstream logic receives realistic payloads even without hardware.
- On each acquisition cycle the camera handler captures or synthesizes an image, the sensor handler pulls Netatmo readings, and the metadata handler augments the payload with timestamps, GPS coordinates, motion hints, and resource flags sourced from configuration.
- `edge_data_collector.formatter.format_data` normalizes motion/resource hints, injects camera identifiers, converts the image to base64-encoded JPEG, and merges any caller-provided metadata. The result matches the schema expected by the downstream FSM (`image_data`, `sensor_data`, `metadata`).
- When `USE_MQTT` is enabled, `edge_data_sender.transmission.mqtt_handler.MqttHandler` publishes the serialized payload to the shared broker/topic; otherwise the system prints the payload for inspection.
- Support scripts manage Netatmo OAuth token refresh, persist updated credentials back to `.env`, and expose utilities to enumerate available sensors. Captured images (real or simulated) are stored under `edge_data_collector/camera/images` for debugging or replay.

## 2. Processing Pi (Coordinator & FSM Orchestrator)
- Runs `main_final.py`, which wires `MQTTReceiver` into `process_message`, validates each payload with `DataValidator`, and persists raw inputs via `StorageManager` for replay or auditing. All messages are tagged with derived filenames so downstream inference can reference consistent artifacts.
- Builds an `FSMStrategy` that converts messages into `FrameContext` objects and forwards them to `FloodFSM`, ensuring sensor metadata, timestamps, motion hints, and resource flags are normalized before inference. Missing baselines are enriched via the shared `BaselineCalculator` if historical data is available.
- Inside the FSM, `ModelManager` keeps lightweight tiers cached locally, but consults `InferenceDispatcher` to decide whether to execute on-device or offload. The dispatcher evaluates three signals: routing policy (`config.INFERENCE_ROUTING`), requested tier vs `LOCAL_YOLO_TIER`, and LLM requirements. When offloading, it builds a request envelope (`task`, `tier`, image data, sensor metadata, unique job ID), publishes it to `MQTT_INFERENCE_REQUEST_TOPIC`, and waits synchronously for a response on `inference/response/<job-id>` with the configured timeout. Failures automatically fall back to the local model path.
- When local execution is chosen, `MultiModelInference` (if available) runs every configured YOLO checkpoint for the requested tier and fuses detections before they are classified by `ClassifierBoth` alongside sensor features. The FSM tracks cooldowns between tier switches and inserts model provenance into each detection to maintain downstream traceability.
- Ambiguous or oscillating states trigger optional LLM confirmation: the FSM invokes `dispatcher.run_remote_llm`, throttles repeat calls with `_llm_last_used`, and merges remote predictions back into state transitions. Returned latency and request IDs are persisted in the `FrameDecision` for observability.
- After each frame, the Pi records the resulting `FrameDecision`, updates baseline statistics through `BaselineCalculator`, and exposes backend metadata (local vs remote, latency, request IDs) so operators can verify offload frequency and performance. All artefacts can be replayed to duplicate decisions offline.

## 3. Processing Jetson (Remote Heavy Inference Worker)
- Runs `jetson_worker/worker.py`, which connects to the broker with a dedicated MQTT client, subscribes to inference request topics, and maintains heartbeats on `inference/jetson/status` so the Pi can detect outages quickly. Each incoming message is parsed, validated, and routed through task-specific handlers (`run_yolo_inference`, `run_llm_inference`).
- Caches YOLO models per tier in `_YOLO_MODELS`; the first time a tier is requested it discovers all `best*.pt` checkpoints relative to the tier directory, loads them with `YOLOv8Inference`, and reuses the instances across jobs. Subsequent requests reuse the cached list, keeping GPU warm and reducing disk thrash.
- For each YOLO task, the worker preprocesses the image, executes every cached model concurrently (serial execution by default, but designed so the loop can be parallelized), aggregates detections with `YOLOv8FinalClassifier`, and formats results (including the `model_ids` provenance) before publishing them back to `inference/response/<job-id>`. The payload mirrors the Pi’s expectations, carrying sensor metadata to keep the fusion pipeline consistent after offload.
- LLM tasks share the same pipeline guard but reuse a thread-safe singleton `LLMImageClassifier`; failures are captured and returned in the response payload so the Pi can fall back to local logic if needed. The Jetson can thus serve both confirmation requests and multi-tier YOLO inference without restarting the worker.
- All MQTT replies include the original job ID, QoS guarantees, and backend metadata such as inference latency, allowing the Processing Pi to reconcile outstanding requests and monitor remote performance. Errors are surfaced as strings in the payload, prompting the Pi to retry locally or flag the event.
- Thread locks guard model and LLM caches, while the worker’s main loop emits periodic status beacons and gracefully handles shutdown signals (`SIGINT`, `SIGTERM`) to avoid orphaned MQTT sessions. Combined with the dispatcher timeout on the Pi, this ensures resilient offloading even across intermittent network conditions.

Together, these components form a resilient, latency-aware pipeline: the
Gathering Pi streams observations, the Processing Pi orchestrates decision logic
and offload policies, and the Jetson delivers heavyweight inference on demand.
