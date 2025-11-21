# config_ablation2.py
# Configuration for Ablation 2: Vision-Only + FSM + Multi-Model + Offload
# - FSM enabled, multi-model consensus, no sensor fusion, allows remote offload

from pathlib import Path
from platform_utils import PLATFORM_YOLO_TIER

PROJECT_ROOT = Path(__file__).resolve().parent
DETECTION_OUTPUT_DIR = PROJECT_ROOT / "storage" / "image-detections"

# ABLATION 2: Use FSM mode
CLASSIFICATION_MODE = "fsm"

# Disable LLM confirmation
USE_LLM_CONFIRMATION = False

# Disable internal energy tracking (using external HMC)
ENABLE_FSM_ENERGY_TRACKING = False

# ABLATION 2: Disable sensor fusion for vision-only testing
DISABLE_SENSOR_FUSION = True

# Image size for inference (width, height)
IMAGE_SIZE = (640, 640)
IMAGE_NAME = "flood-test-ub3"

# ABLATION 2: Multi-model consensus with nano models
model_size = "nano"
model_number = 3

# Importance weights for dynamic model selection
IMPORTANCE = {
    "energy": 0.33,
    "timeliness": 0.33,
    "accuracy": 0.34,
}

# MQTT settings for receiving from collector
MQTT_BROKER_URL = "192.168.20.1"  # Jetson's IP on processor subnet
MQTT_BROKER_PORT = 1883
MQTT_TOPIC = "sensor/data"

# Test dataset settings
TEST_DATASET_DIR = "test_dataset"
IMAGE_MODE = "test_image"
TEST_MOTION = "stop"
TEST_RESOURCE_CONSTRAINED = False

# Baselines (not used since sensor fusion is disabled)
USE_DEFAULT_BASELINE = False
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

# MQTT inference settings for remote offload
MQTT_INFERENCE_REQUEST_TOPIC = "inference/request"
MQTT_INFERENCE_RESPONSE_TOPIC = "inference/response"
MQTT_INFERENCE_HEARTBEAT_TOPIC = "inference/jetson/status"
MQTT_INFERENCE_QOS = 1
MQTT_INFERENCE_TIMEOUT = 6.0
MQTT_INFERENCE_MAX_PAYLOAD = 512_000

# ABLATION 2: Allow remote inference for S1/S2/S3 (default routing)
INFERENCE_ROUTING = {
    "S0": "local",
    "S1": "remote",
    "S2": "remote",
    "S3": "remote",
    "S5": "local",
    "default": "remote",
}

# ABLATION 2: Platform-specific tier (nano on Pi, small on Jetson)
LOCAL_YOLO_TIER = PLATFORM_YOLO_TIER

# Local LLM Configuration (not used)
USE_LOCAL_LLM = False
LOCAL_LLM_BACKEND = "moondream"
LOCAL_LLM_MODEL = str(PROJECT_ROOT / "jetson_worker" / "llm" / "models" / "moondream2")
LOCAL_LLM_USE_4BIT = False
LOCAL_LLM_DEVICE = 'cuda'

