# Flood Detection System

This repository contains an end-to-end prototype for detecting flood conditions by combining computer vision with sensor data. The system implements a Finite State Machine (FSM) for adaptive model selection, sensor fusion with diurnal baselines, and distributed inference across Raspberry Pi and NVIDIA Jetson devices.

## Project Structure

```
cluster_data_receiver/   # MQTT receiver, validation and storage helpers
flood_classifier/        # FSM-based flood classifiers with sensor fusion
yolov8_processor/        # YOLOv8 inference pipeline and model management
jetson_worker/           # Distributed inference worker for Jetson
evaluation_analysis/     # Experiment analysis and metrics computation
tests/                   # Unit, integration and FSM tests
main.py                  # Quick test with simulated inputs
main_test.py             # Loop over the test dataset
main_final.py            # Production MQTT-based operation
main_ablation*.py        # Ablation study entry points (10 configurations)
config.py                # Main configuration file
config_ablation*.py      # Ablation-specific configurations
storage/                 # Saved data, results and baselines
test_dataset/            # Sample evaluation dataset
```

## Installation

Install Python dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

The YOLO models should be placed in `yolov8_processor/model/<tier>/` directories where tier is one of: `nano`, `small`, `medium`, `large`.

## Running the System

### 1. Quick Test (`main.py`)
Demonstrates the full pipeline on a single simulated message:

```bash
python main.py
```

### 2. Dataset Evaluation (`main_test.py`)
Iterates over the files inside `test_dataset/`:

```bash
python main_test.py
```

### 3. Real-Time Operation (`main_final.py`)
Production deployment with MQTT receiver:

```bash
python main_final.py
```

### 4. Ablation Studies
Run specific ablation configurations:

```bash
python main_ablation1.py   # Single-model baseline
python main_ablation4.py   # Full production system
# See ABLATION_STUDY_GUIDE.md for complete list
```

## Configuration

All settings are defined in `config.py`. Key options:

- `CLASSIFICATION_MODE`: `"fsm"` (default), `"yolo_sensor"`, or `"llm_only"`
- `IMAGE_SIZE`: Inference resolution (default: `(640, 640)`)
- `MQTT_BROKER_URL`: MQTT broker address
- `INFERENCE_ROUTING`: FSM state to backend mapping
- `LOCAL_YOLO_TIER`: Platform-specific (nano on Pi, small on Jetson)

### Distributed Inference (Pi ↔ Jetson)

The system supports offloading inference to a Jetson over MQTT. The Jetson hosts the MQTT broker (typically Mosquitto), so both Pis set `MQTT_BROKER_URL` to the Jetson's IP or hostname. Key settings:

- `INFERENCE_ROUTING` controls which FSM states run locally (`S0`, `S5`) and which trigger remote execution (default for `S1`–`S3`).
- `LOCAL_YOLO_TIER` determines the heaviest YOLO model tier kept on the Pi. Larger tiers automatically route to the Jetson.
- `MQTT_INFERENCE_*` settings specify request/response topics, QoS and timeouts for the synchronous MQTT exchange.

## Jetson Worker

The `jetson_worker/worker.py` implements a MQTT-based inference service that:

1. Subscribes to `inference/request` and publishes results to `inference/response/<job-id>` using QoS 1.
2. Accepts JSON payloads with fields `{ "task": "yolo" | "llm", "tier": "small", "image_b64": "...", "metadata": {...} }`.
3. For `task == "yolo"`, loads all available YOLOv8 models for the requested tier, runs inference on each, aggregates the results, and returns `{"detections": [...], "id": <job-id>}`.
4. For `task == "llm"`, invokes the configured local VLM, returning `{ "prediction": 0|1|2, "id": <job-id> }`.
5. Publishes heartbeat messages on `inference/jetson/status` every few seconds so the Pi can detect availability.
6. Logs request IDs, processing time and any errors; on failure, returns `{ "error": "message", "id": <job-id> }` so the Pi can fall back gracefully.

Run the worker on Jetson:

```bash
python jetson_worker/worker.py
```

## Output

Results are stored under `storage/`:
- `storage/data/` — Raw validated payloads
- `storage/data_results/` — Classification results with timing metadata
- `storage/sensor_baselines.json` — Computed diurnal sensor baselines

## Test Dataset

A miniature dataset is included in `test_dataset/` for quick experimentation. The README inside describes how the images and sensor values were generated.

## Documentation

- `ABLATION_STUDY_GUIDE.md` — Comprehensive ablation study guide (10 configurations)
- `system_overview.md` — Detailed architecture documentation
- `setup_guide.md` — Hardware setup instructions
- `PLATFORM_CONFIG_GUIDE.md` — Platform-specific configuration
- `evaluation_protocol.md` — Evaluation methodology
