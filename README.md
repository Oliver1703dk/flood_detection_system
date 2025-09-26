# Flood Detection System

This repository contains an end-to-end prototype for detecting flood conditions by combining computer vision with sensor data. The code is organized into several modules that handle data ingestion, preprocessing, model inference and result storage.

## Project Structure

```
cluster_data_receiver/   # MQTT receiver, validation and storage helpers
flood_classifier/        # Sensor and image-based flood classifiers
main.py                  # Quick test with simulated inputs
main_test.py             # Loop over the test dataset
main_final.py            # Final program for MQTT-based operation
storage/                 # Saved data and baselines
test_dataset/            # Small sample dataset
```

## Installation

Install Python dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

The models used by the YOLOv8 and classifier components should be placed in the directories referenced in `config.py`.

## Running the System

### 1. Quick Test (`main.py`)
`main.py` demonstrates the full pipeline on a single simulated message. The script creates a test image and sensor packet, processes it through the detection models and prints the prediction. Use this to verify that the pipeline is working end to end.

```bash
python main.py
```

### 2. Dataset Evaluation (`main_test.py`)
`main_test.py` iterates over the files inside `test_dataset/`. Each sample contains an image and a JSON label with synthetic sensor readings. The script validates the data, runs YOLOv8, classifies the flood level and prints the accuracy across the dataset.

```bash
python main_test.py
```

### 3. Real-Time Operation (`main_final.py`)
`main_final.py` is intended for deployment. It starts an `MQTTReceiver` that waits for messages on the topic configured in `config.py`. When a message arrives, `process_message` runs the complete pipeline – validation, storage, model inference and result saving.

```bash
python main_final.py
```

The program will keep running until interrupted, printing the prediction for each received message.

## Configuration

All tunable settings such as image size, classification mode and MQTT broker details are defined in `config.py`. Adjust these values to match your hardware setup and network environment.

### Distributed Inference (Pi ↔ Jetson)

The Pi can now offload heavy computer vision or LLM inference tasks to a Jetson over MQTT. Key settings live near the bottom of `config.py`:

- `INFERENCE_ROUTING` controls which FSM states run locally (`S0`, `S5`) and which trigger remote execution (default for `S1`–`S3`).
- `LOCAL_YOLO_TIER` defines the heaviest YOLO model tier that remains on the Pi (`"nano"` by default). Larger tiers automatically route to the Jetson even if the FSM state is local.
- `MQTT_INFERENCE_*` settings specify request/response topics, QoS and timeouts for the synchronous MQTT exchange.

The Jetson should run a small worker service that subscribes to `inference/request`, performs the requested task (YOLO or LLM) and publishes the result on `inference/response/<job-id>`. See the “Jetson Worker Outline” section below for implementation guidance.

## Test Dataset

A miniature dataset is included in the `test_dataset/` folder for quick experimentation. The README inside describes how the images and sensor values were generated and provides a baseline sensor profile used for the synthetic samples.

## Output

Processed data and prediction results are stored under the `storage/` directory. Sensor baselines used by the classifiers are also kept here in `storage/sensor_baselines.json`.

---

This project is a work in progress aimed at exploring multimodal flood detection. Contributions and issue reports are welcome.

## Jetson Worker Outline

The Jetson service should be a lightweight Python process that:

1. Connects to the same MQTT broker and subscribes to `inference/request` and publishes to `inference/response/<job-id>` using QoS 1.
2. Accepts JSON payloads with fields `{ "task": "yolo" | "llm", "tier": "small", "image_b64": "...", "metadata": {...} }`.
3. For `task == "yolo"`, loads all available YOLOv8 models for the requested tier (TensorRT preferable), runs inference on each, aggregates the results using the same logic as the Pi, and returns `{"detections": [...], "id": <job-id>}` with the same detection schema produced by `ResultFormatter`.
4. For `task == "llm"`, invokes the configured LLM or vision model, returning `{ "prediction": 0|1|2, "id": <job-id> }`.
5. Publishes heartbeat messages on `inference/jetson/status` every few seconds so the Pi can detect availability.
6. Logs request IDs, processing time and any errors; on failure, return `{ "error": "message", "id": <job-id> }` so the Pi can fall back gracefully.
