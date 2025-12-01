# config_ablation2b.py
# Configuration for Ablation 2b: Single-Model Vision-Only + FSM + Offload
# - FSM enabled, single nano model, no sensor fusion, allows remote offload
# - Tests FSM without multi-model consensus vs config_ablation2 (3x nano)

from pathlib import Path
from platform_utils import PLATFORM_YOLO_TIER

PROJECT_ROOT = Path(__file__).resolve().parent
DETECTION_OUTPUT_DIR = PROJECT_ROOT / "storage" / "image-detections"

# ABLATION 2b: Use FSM mode
CLASSIFICATION_MODE = "fsm"

# Disable LLM confirmation
USE_LLM_CONFIRMATION = False

# Disable internal energy tracking (using external HMC)
ENABLE_FSM_ENERGY_TRACKING = False

# ABLATION 2b: Disable sensor fusion for vision-only testing
DISABLE_SENSOR_FUSION = True

# Image size for inference (width, height)
IMAGE_SIZE = (640, 640)
IMAGE_NAME = "flood-test-ub3"

# ABLATION 2b: Single nano model (tests FSM without multi-model consensus)
model_size = "nano"
model_number = 1

# Use baseline models (trained on all 3 datasets) for single-model ablations
USE_BASELINE_MODELS = True

# Importance weights for dynamic model selection

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

# FSM parameters
FSM_DEFAULTS = {
    "threshold_low": 0.3,
    "threshold_high": 0.5,
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

# ABLATION 2b: Allow remote inference for S1/S2/S3 (default routing)
INFERENCE_ROUTING = {
    "S0": "local",
    "S1": "remote",
    "S2": "remote",
    "S3": "remote",
    "S5": "local",
    "default": "remote",
}

# When True, all YOLO inference tasks are offloaded to Jetson when motion is FAST,
# regardless of tier or FSM state. This only affects routing decisions, not tier selection.
ALWAYS_OFFLOAD_ON_FAST_MOTION = False

# ABLATION 2b: Platform-specific tier (nano on Pi, small on Jetson)
LOCAL_YOLO_TIER = PLATFORM_YOLO_TIER

# Pre-warm models at startup to avoid cold-start delays
PREWARM_MODELS = True

# Storage directories for results (separate per ablation)
RESULTS_STORAGE_DIR = "storage/data_results/ablation2b"
RESULTS_VIDEO_STORAGE_DIR = "storage/video_results/ablation2b"

# Local LLM Configuration (not used)
USE_LOCAL_LLM = False
LOCAL_LLM_BACKEND = "moondream"
LOCAL_LLM_MODEL = str(PROJECT_ROOT / "jetson_worker" / "llm" / "models" / "moondream2")
LOCAL_LLM_USE_4BIT = False
LOCAL_LLM_DEVICE = 'cuda'

