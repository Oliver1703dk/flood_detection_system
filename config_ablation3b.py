# config_ablation3b.py
# Configuration for Ablation 3b: Single-Model Full System Local-Only
# - FSM enabled, single nano model, sensor fusion enabled, all local
# - Tests full system with single model vs config_ablation3 (3x nano)

from pathlib import Path
from platform_utils import PLATFORM_YOLO_TIER

PROJECT_ROOT = Path(__file__).resolve().parent
DETECTION_OUTPUT_DIR = PROJECT_ROOT / "storage" / "image-detections"

# ABLATION 3b: Use FSM mode
CLASSIFICATION_MODE = "fsm"

# Disable LLM confirmation
USE_LLM_CONFIRMATION = False

# Disable internal energy tracking (using external HMC)
ENABLE_FSM_ENERGY_TRACKING = False

# ABLATION 3b: Enable sensor fusion
DISABLE_SENSOR_FUSION = False

# Image size for inference (width, height)
IMAGE_SIZE = (640, 640)
IMAGE_NAME = "flood-test-ub3"

# ABLATION 3b: Single nano model
model_size = "nano"
model_number = 1

# Importance weights for dynamic model selection
IMPORTANCE = {
    "energy": 0.33,
    "timeliness": 0.33,
    "accuracy": 0.34,
}

# MQTT settings for receiving from collector
MQTT_BROKER_URL = "localhost"
MQTT_BROKER_PORT = 1883
MQTT_TOPIC = "sensor/data"

# Test dataset settings
TEST_DATASET_DIR = "test_dataset"
IMAGE_MODE = "test_image"
TEST_MOTION = "stop"
TEST_RESOURCE_CONSTRAINED = False

# ABLATION 3b: Enable baselines for sensor fusion
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

# MQTT inference settings
MQTT_INFERENCE_REQUEST_TOPIC = "inference/request"
MQTT_INFERENCE_RESPONSE_TOPIC = "inference/response"
MQTT_INFERENCE_HEARTBEAT_TOPIC = "inference/jetson/status"
MQTT_INFERENCE_QOS = 1
MQTT_INFERENCE_TIMEOUT = 6.0
MQTT_INFERENCE_MAX_PAYLOAD = 512_000

# ABLATION 3b: Force everything local
INFERENCE_ROUTING = {
    "S0": "local",
    "S1": "local",
    "S2": "local",
    "S3": "local",
    "S5": "local",
    "default": "local",
}

# ABLATION 3b: Platform-specific tier (nano on Pi, small on Jetson)
LOCAL_YOLO_TIER = PLATFORM_YOLO_TIER

# Local LLM Configuration (not used)
USE_LOCAL_LLM = False
LOCAL_LLM_BACKEND = "moondream"
LOCAL_LLM_MODEL = str(PROJECT_ROOT / "jetson_worker" / "llm" / "models" / "moondream2")
LOCAL_LLM_USE_4BIT = False
LOCAL_LLM_DEVICE = 'cuda'

