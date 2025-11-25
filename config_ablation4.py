# config_ablation4.py
# Configuration for Ablation 4: Full System Remote-Enabled (Production)
# - FSM enabled, multi-model consensus, sensor fusion enabled, remote offload enabled

from pathlib import Path
from platform_utils import PLATFORM_YOLO_TIER

PROJECT_ROOT = Path(__file__).resolve().parent
DETECTION_OUTPUT_DIR = PROJECT_ROOT / "storage" / "image-detections"

# ABLATION 4: Use FSM mode
CLASSIFICATION_MODE = "fsm"

# Disable LLM confirmation (can be enabled for VLM testing)
USE_LLM_CONFIRMATION = False

# Disable internal energy tracking (using external HMC)
ENABLE_FSM_ENERGY_TRACKING = False

# ABLATION 4: Enable sensor fusion
DISABLE_SENSOR_FUSION = False

# Image size for inference (width, height)
IMAGE_SIZE = (640, 640)
IMAGE_NAME = "flood-test-ub3"

# ABLATION 4: Multi-model consensus with nano models
model_size = "nano"
model_number = 3


# MQTT settings for receiving from collector
MQTT_BROKER_URL = "192.168.20.1"  # Jetson's IP on processor subnet
MQTT_BROKER_PORT = 1883
MQTT_TOPIC = "sensor/data"

# Test dataset settings
TEST_DATASET_DIR = "test_dataset"
IMAGE_MODE = "test_image"
TEST_MOTION = "stop"
TEST_RESOURCE_CONSTRAINED = False

# ABLATION 4: Enable baselines for sensor fusion
USE_DEFAULT_BASELINE = True
DEFAULT_BASELINE = {
    "pre_dawn":  { "temperature_baseline": 17.0, "humidity_baseline": 78.0, "pressure_baseline": 1016.0 },
    "midday":    { "temperature_baseline": 17.0, "humidity_baseline": 78.0, "pressure_baseline": 1016.0 },
    "evening":   { "temperature_baseline": 17.0, "humidity_baseline": 78.0, "pressure_baseline": 1016.0 },
    "night":     { "temperature_baseline": 17.0, "humidity_baseline": 78.0, "pressure_baseline": 1016.0 },
}

# FSM parameters
FSM_DEFAULTS = {
    "threshold_low": 0.35,
    "threshold_high": 0.65,
    "frames_high": 2,
    "frames_low": 2,
    "frames_ambiguous": 2,
    "model_cooldown": 1,
    "flap_window": 8,
    "resource_skip_ratio": 3,
    "llm_model": "gpt-4.1-mini",
}

FSM_REPEAT_COUNT = 5

# Disable baseline updates during evaluation (too slow - runs every frame)
ENABLE_BASELINE_UPDATES = False

# MQTT inference settings for remote offload
MQTT_INFERENCE_REQUEST_TOPIC = "inference/request"
MQTT_INFERENCE_RESPONSE_TOPIC = "inference/response"
MQTT_INFERENCE_HEARTBEAT_TOPIC = "inference/jetson/status"
MQTT_INFERENCE_QOS = 1
MQTT_INFERENCE_TIMEOUT = 6.0
MQTT_INFERENCE_MAX_PAYLOAD = 512_000

# ABLATION 4: Enable remote inference for higher states (production config)
INFERENCE_ROUTING = {
    "S0": "local",
    "S1": "remote",
    "S2": "remote",
    "S3": "remote",
    "S5": "local",
    "default": "remote",
}

# ABLATION 4: Tiers above small go to Jetson
# ABLATION 4: Platform-specific tier (nano on Pi, small on Jetson)
LOCAL_YOLO_TIER = PLATFORM_YOLO_TIER

# Pre-warm models at startup to avoid cold-start delays
PREWARM_MODELS = True

# Storage directories for results (separate per ablation)
RESULTS_STORAGE_DIR = "storage/data_results/ablation4"
RESULTS_VIDEO_STORAGE_DIR = "storage/video_results/ablation4"

# Local LLM Configuration (can be enabled for testing)
USE_LOCAL_LLM = False
LOCAL_LLM_BACKEND = "moondream"
LOCAL_LLM_MODEL = str(PROJECT_ROOT / "jetson_worker" / "llm" / "models" / "moondream2")
LOCAL_LLM_USE_4BIT = False
LOCAL_LLM_DEVICE = 'cuda'

