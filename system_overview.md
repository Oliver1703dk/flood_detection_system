# Flood Detection System Overview

This document summarizes the production architecture for the distributed flood
detection system spanning two Raspberry Pi devices and an NVIDIA Jetson. The
components coordinate via MQTT messaging and share a common storage layout and
configuration surface.

## 1. Gathering Pi (Sensor & Image Capture)
- Runs `main_final.py` in collector mode to refresh `.env` secrets, read feature flags from `config.py`, and instantiate camera, Netatmo sensor (or simulator), metadata enrichment, and MQTT handlers.
- Netatmo integration captures barometric pressure, temperature, humidity, CO2, and rainfall metrics; simulation mode produces bounded synthetic readings so downstream logic always receives realistic payloads.
- Each acquisition cycle captures or synthesizes an image, gathers sensor readings, and enriches metadata with timestamps, GPS, motion hints, and resource flags.
- `edge_data_collector.formatter.format_data` normalizes metadata, injects camera identifiers, encodes imagery as base64 JPEG, and packages the payload as `{image_data, sensor_data, metadata}` for the Processing Pi.
- When `USE_MQTT` is enabled, `edge_data_sender.transmission.mqtt_handler.MqttHandler` publishes the payload to the shared broker/topic; otherwise the payload prints for manual inspection.
- Support scripts refresh Netatmo OAuth tokens, persist credentials to `.env`, and retain captured images under `edge_data_collector/camera/images` for replay.

## 2. Processing Pi: Core Data Pipeline and Decision Logic
The Processing Pi consumes the Gathering Pi stream, normalizes inputs, drives the finite-state decision engine, and persists results.

- **Payload Ingestion and Preparation**: `MQTTReceiver` buffers incoming payloads, `DataValidator` enforces the expected schema, and `StorageManager` archives raw JSON blobs under `storage/data/`. Only the freshest frame is retained via `LatestPayloadBuffer` so downstream stages operate on current context.
- **Data Normalization (FSM Strategy)**: The production pipeline always runs the FSM strategy, transforming validated payloads into a normalized frame context with harmonized timestamps, motion/resource hints, sensor values, and base64 imagery.
- **State Orchestration (Finite-State Machine)**:
  - *State Set*: S0 – baseline watch; S1 – watchful suspicion; S2 – elevated review with sensor corroboration; S3 – confirmed flood response; S5 – resource-conserving fallback.
  - *Process Flow*: 1) baseline retrieval; 2) tier selection; 3) routing choice (local vs remote); 4) inference execution; 5) scoring fusion; 6) conflict detection; 7) LLM trigger when ambiguity persists; 8) state transition update; 9) decision output for downstream consumers.
- **YOLO Inference Flow**:
  1. Preprocessing: decode the base64 image, resize to the configured `IMAGE_SIZE`, and normalize channels.
  2. Multi-model inference: execute the configured checkpoints for the selected tier, using cached models when available.
  3. Aggregation: merge detections across checkpoints, propagate model provenance, and align detections with sensor context.
  4. Formatting: emit normalized detection objects ready for fusion and persistence.
  - *Tier Lineup*: Nano – emergency-only minimal load; Small – Pi-resident default; Medium – remote precision tier; Large – heavy Jetson tier for complex frames.
- **Baseline Calculation**: The baseline calculator scans `storage/data_results/`, groups historical sensor readings into diurnal windows (pre-dawn, midday, evening, night), computes exponentially weighted moving averages, and persists the consolidated values to `storage/sensor_baselines.json`.
- **Scoring Logic**:
  1. Retrieve the current baseline snapshot for the frame window.
  2. Compute deltas between live sensor data and baseline.
  3. Apply gating thresholds to suppress noise and respect cooldowns.
  4. Calculate `sensor_boost` from anomaly magnitude and direction.
  5. Calculate `image_score` from YOLO detections and tier confidence.
  6. Add a consensus bonus when sensor and image signals align.
  7. Combine image, sensor, and consensus terms into a unified score.
  8. Classify the frame into no-flood, watch, or flood states for FSM consumption.
- **Inference Routing and Offload**:
  - Routing criteria: required tier, per-state routing policy, Jetson health check, and LLM confirmation requirement.
  - Remote YOLO Flow: 1) package image, sensor, and baseline context; 2) publish to `inference/request`; 3) await Jetson response topic; 4) merge results back into the FSM pipeline.
  - Remote LLM Flow: 1) package ambiguity snapshot and imagery; 2) request Jetson VLM confirmation; 3) receive classification with explanation metadata; 4) feed the verdict into scoring.
- **Result Persistence**: `DataResultsSaver` writes enriched decisions to `storage/data_results/`, capturing image metadata, sensor deltas, FSM state, model tier usage, backend provenance, scores, and final classification for each frame.

## 3. Jetson Worker (Remote Inference Node)
The Jetson worker provides heavyweight YOLO and local VLM inference for frames escalated by the Processing Pi.

- **Connectivity and Queuing**: The worker subscribes to MQTT request topics, advertises availability through a retained heartbeat every 10 seconds, and manages two task types (`yolo` and `llm`). A single-job queue retains only the latest request, superseding older jobs when new payloads arrive.
- **YOLO Processing Pipeline**:
  1. Preprocessing: decode base64 imagery, resize, and normalize using the shared image processor.
  2. Model caching: lazily load all checkpoints for the requested tier on first use and reuse them for subsequent frames.
  3. Multi-model inference: run each checkpoint, collecting detections per model.
  4. Aggregation: fuse detections into a consensus result and annotate with model provenance.
  5. Response packaging: return detections, tier metadata, sensor context echo, and any aggregation diagnostics.
- **LLM Integration**: Production relies on the local vision-language model (VILA1.5-3b); the OpenAI API path is retained for testing only. The inference sequence is: 1) decode image bytes; 2) invoke the lightweight classifier facade; 3) build a prompt that blends sensor anomalies with flood cues; 4) run the local VLM inference; 5) respond with the predicted label and supporting details.
  - *Local VLM Details*: Model path `jetson_worker/llm/models/VILA1.5-3b`, 4-bit quantization enabled, and lazy loading that initializes weights on the first LLM job or during optional preload at startup.
- **Response Handling**: Each MQTT response includes the job id, tier, detections or VLM verdict, sensor echoes, latency metrics, and any error flags for the Processing Pi to ingest.
- **Reliability Measures**: Single-job queue prevents backlog buildup, heartbeats expose liveness to the Pi, and graceful degradation routes errors back to the FSM so it can fall back to local processing.

## 4. Data Flow Summary
- End-to-end pipeline:
  1. Gathering Pi captures image and sensor snapshot.
  2. Gathering Pi publishes the normalized payload over MQTT.
  3. Processing Pi ingests, validates, and archives the raw frame.
  4. FSM normalizes the frame context and retrieves baselines.
  5. FSM selects a model tier and routing policy.
  6. Local or remote YOLO inference produces detections.
  7. FSM computes scores, resolves conflicts, and triggers LLM confirmation when needed.
  8. Remote VLM (if invoked) returns a consensus label.
  9. FSM updates state, finalizes the decision, and persists results.
  10. Decisions propagate to downstream monitoring and alerting.
- MQTT topics in use: `sensor/data`, `inference/request`, `inference/response`, `inference/jetson/status`.

## 5. Storage Structure
```
storage/
├── data/
│   └── <camera_id>_<YYYYMMDD_HHMMSSmmm>.json
├── data_results/
│   └── <YYYY-MM-DD>/
│       └── <camera_id>_<HH-MM-SS>.json
└── sensor_baselines.json
```

Example `storage/data_results/<date>/<camera>_<time>.json`:
```json
{
  "classification_result": {
    "prediction": 2,
    "state": "flood",
    "model_tier": "medium",
    "scores": {
      "combined_score": 2.68,
      "image_score": 2.68,
      "sensor_boost": 0.0
    },
    "backend": "remote",
    "conflict": true,
    "counters": {
      "ambiguous": 3,
      "conflict": 3,
      "high": 3,
      "low": 0
    }
  },
  "metadata": {
    "camera_id": "CAM123",
    "timestamp": "2025-02-07T12:00:00Z",
    "sensor_baseline": {
      "temperature_baseline": 22.0,
      "humidity_baseline": 50.0,
      "pressure_baseline": 1015.0
    },
    "sensor_anomalies": {
      "delta_temperature": 3.0,
      "delta_humidity": 5.0,
      "delta_pressure": -1.75
    },
    "motion": "stop"
  },
  "sensor_data": {
    "temperature": 25.0,
    "humidity": 55.0,
    "pressure": 1013.25
  }
}
```

## 6. Configuration
- Production defaults from `config.py`:
  - `CLASSIFICATION_MODE = "fsm"` keeps the finite-state strategy active.
  - `USE_LLM_CONFIRMATION = True` enables VLM arbitration when ambiguity persists.
  - `IMAGE_SIZE = (640, 640)` aligns preprocessing across Pi and Jetson.
  - `LOCAL_YOLO_TIER = "small"` determines the highest tier retained locally.
  - `IMPORTANCE = {"energy": 0.33, "timeliness": 0.33, "accuracy": 0.34}` biases tier selection.
  - `MQTT_TOPIC = "sensor/data"` and inference topics `inference/request`, `inference/response`, `inference/jetson/status` coordinate messaging.
  - `MQTT_INFERENCE_TIMEOUT = 6.0` seconds bounds remote call latency.
  - `USE_LOCAL_LLM = True`, `LOCAL_LLM_MODEL = jetson_worker/llm/models/VILA1.5-3b`, `LOCAL_LLM_USE_4BIT = True`, and `LOCAL_LLM_DEVICE = "cuda"` ensure the Jetson VLM runs locally.
- Model path structure:
```
yolov8_processor/model/
├── nano/
│   └── best*.pt
├── small/
│   └── best*.pt
├── medium/
│   └── best*.pt
└── large/
    └── best*.pt
```
These directories house the checkpoints used by the multi-tier inference pipeline, with filenames following the `best<number>.pt` convention for each tier.
