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

## Test Dataset

A miniature dataset is included in the `test_dataset/` folder for quick experimentation. The README inside describes how the images and sensor values were generated and provides a baseline sensor profile used for the synthetic samples.

## Output

Processed data and prediction results are stored under the `storage/` directory. Sensor baselines used by the classifiers are also kept here in `storage/sensor_baselines.json`.

---

This project is a work in progress aimed at exploring multimodal flood detection. Contributions and issue reports are welcome.
