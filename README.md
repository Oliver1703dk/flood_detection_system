# Flood Detection System

This repository contains the **Processing Pi + Jetson Worker** code for an edge-based flood/standing-water detection system. It combines computer vision (YOLOv8 ensembles) with environmental sensors and a finite state machine (FSM) to adapt model tiers, offload heavy inference to a Jetson, and fuse diurnal sensor baselines.

This codebase corresponds to the **Processing/Jetson side** of the replication package used in the ICSA 2026 paper *“Edge-Based Standing-Water Detection via FSM-Guided Tiering and Multi-Model Consensus”*. The companion **Gathering Pi** code (camera + sensors) lives in the [`edge_data_collector`](https://github.com/Oliver1703dk/edge_data_collector) repository.

## Project Structure

```text
cluster_data_receiver/   # MQTT receiver, validation and storage helpers
flood_classifier/        # FSM-based flood classifiers with sensor fusion
yolov8_processor/        # YOLOv8 inference pipeline and model management
jetson_worker/           # Distributed inference worker for Jetson
evaluation_analysis/     # Experiment analysis and metrics computation
tests/                   # Unit, integration and FSM tests
main.py                  # Quick test with simulated inputs
main_test.py             # Loop over the test dataset
main_final.py            # Production-style MQTT-based operation (ID 4 in the paper)
main_ablation*.py        # Ablation study entry points (10 configurations)
config.py                # Main configuration file
config_ablation*.py      # Ablation-specific configurations
storage/                 # Saved data, results and baselines
test_dataset/            # Sample evaluation dataset
```

For details on how `main_ablation*.py` map to the ablation IDs in Table IV of the paper (1, 1b, 2, 2b, 3, 3b, 4, 4b, 5, 6), see **`ABLATION_STUDY_GUIDE.md`**.

## Installation

Install Python dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

YOLOv8 model weights should be placed under:

```text
yolov8_processor/model/<tier>/
```

where `<tier>` is one of: `nano`, `small`, `medium`, `large`.

## Running the System

### 1. Quick Test (`main.py`)

Demonstrates the full pipeline on a single simulated message (no MQTT):

```bash
python main.py
```

### 2. Dataset Evaluation (`main_test.py`)

Iterates over the files inside `test_dataset/` to run the FSM + scoring pipeline on a small, local dataset:

```bash
python main_test.py
```

### 3. Real-Time MQTT Operation (`main_final.py`)

Runs the Processing Pi side in a production-style configuration with an MQTT receiver. This corresponds to the **“production” configuration (ID 4)** in the paper when used with the appropriate config and Jetson worker.

```bash
python main_final.py
```

### 4. Ablation Studies (`main_ablation*.py`)

Reproduce the ablation configurations from the paper on real hardware. Example:

```bash
python main_ablation1.py   # e.g., static_nano / static_small baseline
python main_ablation4.py   # Production system (ID 4 in Table IV)
# See ABLATION_STUDY_GUIDE.md for the complete list and mapping.
```

Each ablation script loads its own `config_ablation*.py` to fix model tiers, consensus size, sensor fusion, and offload policy.

## Configuration

Most settings are defined in `config.py` (and overridden in `config_ablation*.py` for experiments). Key options include:

* `CLASSIFICATION_MODE`: `"fsm"` (default), `"yolo_sensor"`, or `"llm_only"`

  * Only `"fsm"` and `"yolo_sensor"` are used in the ICSA 2026 experiments; `"llm_only"` is experimental.
* `IMAGE_SIZE`: Inference resolution (default: `(640, 640)`).
* `MQTT_BROKER_URL`: MQTT broker address (typically the Jetson IP/hostname).
* `INFERENCE_ROUTING`: Maps FSM states to local vs. remote tiers.
* `LOCAL_YOLO_TIER`: Heaviest YOLO tier kept on the Processing Pi.

The classification thresholds and FSM parameters (e.g., `threshold_low`, `threshold_high`, hysteresis counters) match the values described in the paper’s decision-logic section and can be tuned in the relevant modules under `flood_classifier/`.

### Distributed Inference (Processing Pi ↔ Jetson)

The system supports offloading heavier tiers to a Jetson via MQTT:

* The **Jetson** usually hosts the MQTT broker (e.g., Mosquitto).
* Both Pis set `MQTT_BROKER_URL` to the Jetson’s IP or hostname.
* `INFERENCE_ROUTING` decides which FSM states run locally and which offload.

By default:

* Light tiers (e.g., `nano`) run locally on the **Processing Pi**.
* Heavy tiers (`small`, `medium`, `large`) are sent to the **Jetson Worker**.

FSM state mapping is consistent with the paper:

* **S0 – Normal Watch**: local nano tier.
* **S1 – Uncertainty Investigation**: escalate (small/medium).
* **S2 – Confirmed Flood**: maintain flood decision with small/large tiers.
* **S3 – Ambiguity / Conflict Resolution**: prefer medium/large tiers.
* **S5 – Resource-Constrained**: local nano-only inference and reduced cadence.

`INFERENCE_ROUTING` allows you to override which tiers are used in which states (e.g., for specific ablations).

## Jetson Worker

The Jetson worker (`jetson_worker/worker.py`) implements an MQTT-based inference service:

1. Subscribes to `inference/request` and publishes results to `inference/response/<job-id>` (QoS 1).

2. Accepts JSON payloads with fields:

   ```json
   {
     "task": "yolo" | "llm",
     "tier": "nano|small|medium|large",
     "image_b64": "...",
     "metadata": { ... }
   }
   ```

3. For `task == "yolo"`:

   * Loads the configured YOLOv8 models for the requested tier.
   * Runs inference on each model.
   * Aggregates detections via the consensus logic (confidence-weighted fusion, agreement counts).
   * Returns:

     ```json
     {
       "detections": [...],
       "id": "<job-id>"
     }
     ```

4. For `task == "llm"`:

   * **Experimental**: invokes a local vision/LLM component if configured.
   * This path is **not used in the ICSA 2026 evaluation results**.

5. Publishes heartbeat messages on `inference/jetson/status` so the Processing Pi can detect availability.

6. Logs job IDs, timing, and errors. On failure it returns:

   ```json
   { "error": "message", "id": "<job-id>" }
   ```

   allowing the Processing Pi to fall back to local nano-tier inference and enter the resource-constrained state (S4) until the Jetson is healthy again.

Run the worker on Jetson:

```bash
python jetson_worker/worker.py
```

## Output Layout

All results are written under `storage/` on the Processing Pi:

* `storage/data/` — Raw, validated payloads (input frames + sensor data).
* `storage/data_results/` — Per-frame classification results + timing + FSM state (used to compute the tables/figures in the paper).
* `storage/sensor_baselines.json` — Diurnal EWMA sensor baselines (four daily windows).

These logs are deterministic for a given input stream and configuration, enabling hardware-in-the-loop replays and controlled ablations as described in the paper.

## Test Dataset

A small evaluation dataset is included in:

```text
test_dataset/
```

This is intended for quick experiments and sanity checks. The README inside `test_dataset/` explains how the images and sensor values were generated and how they relate to the sequences in the paper.

For full reproduction of the paper’s tables and figures, use the **field video replays + sensor variants** and the corresponding ablation scripts as described in `evaluation_protocol.md` and `ABLATION_STUDY_GUIDE.md`.

## Documentation

Additional documentation files:

* `ABLATION_STUDY_GUIDE.md` — Mapping between `main_ablation*.py` configs and the ablation IDs in the paper; how to run each configuration.
* `system_overview.md` — Architectural overview (Processing Pi, Jetson Worker, FSM, consensus, sensor fusion).
* `setup_guide.md` — Hardware setup (Pis, Jetson, MQTT broker, camera, sensors).
* `PLATFORM_CONFIG_GUIDE.md` — Platform-specific config hints (RPi vs Jetson variants).
* `evaluation_protocol.md` — How to perform hardware-in-the-loop replays and compute metrics.

Together with the `edge_data_collector` repo, these documents form the **replication package** for the ICSA 2026 paper.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Author

Oliver Larsen, University of Southern Denmark (SDU)
