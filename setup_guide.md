# Flood Detection System Deployment Guide

This guide walks through preparing both the Raspberry Pi (orchestration node) and the NVIDIA Jetson AGX Orin (heavy inference node) for the distributed flood-detection pipeline.

---

## 1. Shared Prerequisites

- Access to the Jetson-hosted MQTT broker (e.g., Mosquitto) reachable by both devices.
- Matching copies of this repository (or the relevant subfolders) on the Pi and Jetson.
- YOLO model weights placed under `yolov8_processor/model/<tier>/best*.pt` on the respective device (nano on the Pi, larger tiers on the Jetson).
- Any required API keys or credentials stored in environment variables (e.g., `OPENAI_API_KEY` for LLM inference).

---

## 2. Raspberry Pi Setup (Coordinator)

1. **System packages**
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip python3-venv
   ```

2. **Python environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Repository configuration**
   - Edit `config.py`:
     - Set `MQTT_BROKER_URL`, `MQTT_BROKER_PORT`, and the inference topics/timeouts so the Pi points to the Jetson-hosted broker.
     - Ensure `LOCAL_YOLO_TIER = "nano"` (default) so the Pi keeps the light model locally.
     - Adjust `INFERENCE_ROUTING` if you need different state-based routing.
   - Confirm the nano weights exist under `yolov8_processor/model/nano/`.

4. **Run the coordinator**
   ```bash
   source .venv/bin/activate
   python main_final.py
   ```
   The script starts the MQTT receiver and orchestrates inference, offloading heavier jobs to the Jetson.

5. **Optional checks**
   - For quick validation: `python main.py` to run a single simulated message.
   - Inspect logs for backend metadata (local vs remote) in the FSM decisions.

---

## 3. Jetson AGX Orin Setup (Inference Worker)

1. **System packages**
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip python3-venv
   ```
   Enable NVIDIA drivers and CUDA/TensorRT as needed for YOLO acceleration.

2. **Python environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   Install any extra libraries required for optimized inference (TensorRT bindings, etc.).

3. **Repository configuration**
   - Mirror the Pi's `config.py` (at least for MQTT settings) but keep `MQTT_BROKER_URL` at `localhost` when the broker runs on this Jetson.
   - Place heavier YOLO weights (e.g., `small`, `medium`, `large`, `xlarge`) under `yolov8_processor/model/<tier>/`.
   - Set environment variables for the LLM if using a hosted API (e.g., `export OPENAI_API_KEY=...`).
   - Install and start Mosquitto (or your chosen broker) so remote Pis can connect to this Jetson instance.

4. **Run the inference worker**
   ```bash
   source .venv/bin/activate
   python jetson_worker/worker.py
   ```
   The worker subscribes to `inference/request`, performs YOLO/LLM inference, and publishes responses with correlation IDs.

5. **Operational tips**
   - Watch broker logs or `mosquitto_sub -t inference/jetson/status` to confirm heartbeats.
   - Use `journalctl` or custom logging to monitor GPU usage and inference latency.

---

## 4. End-to-End Verification

1. Start the Jetson worker and ensure it emits heartbeat messages.
2. Launch the Pi `main_final.py` script.
3. Send a test message (e.g., `python main.py` or publish to the MQTT topic). Confirm:
   - The Pi logs indicate remote inference when the FSM enters states S1–S3 or requests larger YOLO tiers.
   - The Jetson logs show incoming jobs and successful completions.
4. Check saved results under `storage/` on the Pi for classifications and metadata indicating which backend processed each frame.

---

## 5. Troubleshooting

- **No remote inference**: Verify Jetson heartbeats; if absent, check MQTT connectivity and worker logs.
- **Model not found**: Ensure tier paths in `config.py` match the actual filenames under `yolov8_processor/model/` on BOTH devices.
- **LLM errors**: Confirm API credentials, network access, or switch to an on-device model if offline.
- **Timeouts**: Increase `MQTT_INFERENCE_TIMEOUT` or optimize Jetson inference (e.g., TensorRT) to keep job latency within limits.

With both nodes configured, the FSM automatically routes workloads based on state and requested model tier, allowing the Pi to stay responsive while the Jetson handles heavier computation.
