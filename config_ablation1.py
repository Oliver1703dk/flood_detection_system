# config_ablation1.py
# Configuration for Ablation 1: Static Small-YOLO Only (Baseline)
# - Single small model, no FSM, no sensor fusion, no offload
# Note: Evaluation plan specifies "1× small" as the strongest realistic single-model
# baseline that runs at 5-7 fps on Raspberry Pi 5

from pathlib import Path
from platform_utils import PLATFORM_YOLO_TIER

PROJECT_ROOT = Path(__file__).resolve().parent
DETECTION_OUTPUT_DIR = PROJECT_ROOT / "storage" / "image-detections"

# ABLATION 1: Use yolo_sensor mode (not FSM)
CLASSIFICATION_MODE = "yolo_sensor"

# Disable LLM confirmation
USE_LLM_CONFIRMATION = False

# Disable internal energy tracking (using external HMC)
ENABLE_FSM_ENERGY_TRACKING = False

# ABLATION 1: Disable sensor fusion for vision-only baseline
DISABLE_SENSOR_FUSION = True

# Image size for inference (width, height)
IMAGE_SIZE = (640, 640)
IMAGE_NAME = "flood-test-ub3"

# ABLATION 1: Single small model (as per evaluation plan: 1× small)
model_size = "small"
model_number = 1

# Use baseline models (trained on all 3 datasets) for single-model ablations
USE_BASELINE_MODELS = True


# MQTT settings for receiving from collector
MQTT_BROKER_URL = "192.168.42.10"
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
    "threshold_low": 0.12,
    "threshold_high": 0.4,
    "frames_high": 1,
    "frames_low": 1,
    "frames_ambiguous": 1,
    "model_cooldown": 1,
    "flap_window": 8,
    "resource_skip_ratio": 3,
    "llm_model": "gpt-4.1-mini",
}

FSM_REPEAT_COUNT = 5

# Disable baseline updates during evaluation (too slow - runs every frame)
ENABLE_BASELINE_UPDATES = False

# MQTT inference settings (not used - no remote offload in this ablation)
MQTT_INFERENCE_REQUEST_TOPIC = "inference/request"
MQTT_INFERENCE_RESPONSE_TOPIC = "inference/response"
MQTT_INFERENCE_HEARTBEAT_TOPIC = "inference/jetson/status"
MQTT_INFERENCE_QOS = 1
MQTT_INFERENCE_TIMEOUT = 6.0
MQTT_INFERENCE_MAX_PAYLOAD = 512_000

# No remote inference for Ablation 1
INFERENCE_ROUTING = {
    "S0": "local",
    "S1": "local",
    "S2": "local",
    "S3": "local",
    "S5": "local",
    "default": "local",
}

# When True, all YOLO inference tasks are offloaded to Jetson when motion is FAST,
# regardless of tier or FSM state. This only affects routing decisions, not tier selection.
ALWAYS_OFFLOAD_ON_FAST_MOTION = False

LOCAL_YOLO_TIER = "small"

# Pre-warm models at startup to avoid cold-start delays
PREWARM_MODELS = True

# Storage directories for results (separate per ablation)
RESULTS_STORAGE_DIR = "storage/data_results/ablation1"
RESULTS_VIDEO_STORAGE_DIR = "storage/video_results/ablation1"

# Local LLM Configuration (not used)
USE_LOCAL_LLM = False
LOCAL_LLM_BACKEND = "moondream"
LOCAL_LLM_MODEL = str(PROJECT_ROOT / "jetson_worker" / "llm" / "models" / "moondream2")
LOCAL_LLM_USE_4BIT = False
LOCAL_LLM_DEVICE = 'cuda'

