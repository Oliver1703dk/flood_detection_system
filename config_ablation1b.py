# config_ablation1b.py
# Configuration for Ablation 1b: Multi-Model Medium-YOLO Baseline (3x medium)
# - Three medium models with consensus, no FSM, no sensor fusion, no offload
# - Tests multi-model consensus effect vs config_ablation1 (1x medium)

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DETECTION_OUTPUT_DIR = PROJECT_ROOT / "storage" / "image-detections"

# ABLATION 1b: Use yolo_sensor mode (not FSM)
CLASSIFICATION_MODE = "yolo_sensor"

# Disable LLM confirmation
USE_LLM_CONFIRMATION = False

# Disable internal energy tracking (using external HMC)
ENABLE_FSM_ENERGY_TRACKING = False

# ABLATION 1b: Disable sensor fusion for vision-only baseline
DISABLE_SENSOR_FUSION = True

# Image size for inference (width, height)
IMAGE_SIZE = (640, 640)
IMAGE_NAME = "flood-test-ub3"

# ABLATION 1b: Three medium models with consensus
model_size = "medium"
model_number = 3

# Importance weights (not used in yolo_sensor mode, but kept for consistency)
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

# Baselines (not used since sensor fusion is disabled)
USE_DEFAULT_BASELINE = False
DEFAULT_BASELINE = {
    "pre_dawn":  { "temperature_baseline": 17.0, "humidity_baseline": 78.0, "pressure_baseline": 1016.0 },
    "midday":    { "temperature_baseline": 17.0, "humidity_baseline": 78.0, "pressure_baseline": 1016.0 },
    "evening":   { "temperature_baseline": 17.0, "humidity_baseline": 78.0, "pressure_baseline": 1016.0 },
    "night":     { "temperature_baseline": 17.0, "humidity_baseline": 78.0, "pressure_baseline": 1016.0 },
}

# FSM parameters (not used in yolo_sensor mode)
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

# MQTT inference settings (not used - no remote offload in this ablation)
MQTT_INFERENCE_REQUEST_TOPIC = "inference/request"
MQTT_INFERENCE_RESPONSE_TOPIC = "inference/response"
MQTT_INFERENCE_HEARTBEAT_TOPIC = "inference/jetson/status"
MQTT_INFERENCE_QOS = 1
MQTT_INFERENCE_TIMEOUT = 6.0
MQTT_INFERENCE_MAX_PAYLOAD = 512_000

# No remote inference for Ablation 1b
INFERENCE_ROUTING = {
    "S0": "local",
    "S1": "local",
    "S2": "local",
    "S3": "local",
    "S5": "local",
    "default": "local",
}

LOCAL_YOLO_TIER = "medium"

# Local LLM Configuration (not used)
USE_LOCAL_LLM = False
LOCAL_LLM_BACKEND = "moondream"
LOCAL_LLM_MODEL = str(PROJECT_ROOT / "jetson_worker" / "llm" / "models" / "moondream2")
LOCAL_LLM_USE_4BIT = False
LOCAL_LLM_DEVICE = 'cuda'

