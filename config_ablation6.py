# config_ablation6.py
# Configuration for Ablation 6: Always Offload Medium (Naive Baseline)
# - Fixed medium tier, always offload to Jetson, no FSM adaptive logic
# - Shows why blind offloading is terrible for energy (reviewers always ask for this)

from pathlib import Path
from platform_utils import PLATFORM_YOLO_TIER

PROJECT_ROOT = Path(__file__).resolve().parent
DETECTION_OUTPUT_DIR = PROJECT_ROOT / "storage" / "image-detections"

# ABLATION 6: Use FSM mode (needed for remote offload support, but FIXED_TIER makes it static)
CLASSIFICATION_MODE = "fsm"

# Disable LLM confirmation
USE_LLM_CONFIRMATION = False

# Disable internal energy tracking (using external HMC)
ENABLE_FSM_ENERGY_TRACKING = False

# ABLATION 6: Enable sensor fusion (can keep for fair comparison)
DISABLE_SENSOR_FUSION = False

# Image size for inference (width, height)
IMAGE_SIZE = (640, 640)
IMAGE_NAME = "flood-test-ub3"

# ABLATION 6: Single medium model
model_size = "medium"
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

# ABLATION 6: Enable baselines for sensor fusion
USE_DEFAULT_BASELINE = True
DEFAULT_BASELINE = {
    "pre_dawn":  { "temperature_baseline": 17.0, "humidity_baseline": 78.0, "pressure_baseline": 1016.0 },
    "midday":    { "temperature_baseline": 17.0, "humidity_baseline": 78.0, "pressure_baseline": 1016.0 },
    "evening":   { "temperature_baseline": 17.0, "humidity_baseline": 78.0, "pressure_baseline": 1016.0 },
    "night":     { "temperature_baseline": 17.0, "humidity_baseline": 78.0, "pressure_baseline": 1016.0 },
}

# FSM parameters (not really used since FIXED_TIER overrides tier selection)
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

# MQTT inference settings for remote offload
MQTT_INFERENCE_REQUEST_TOPIC = "inference/request"
MQTT_INFERENCE_RESPONSE_TOPIC = "inference/response"
MQTT_INFERENCE_HEARTBEAT_TOPIC = "inference/jetson/status"
MQTT_INFERENCE_QOS = 1
MQTT_INFERENCE_TIMEOUT = 6.0
MQTT_INFERENCE_MAX_PAYLOAD = 512_000

# ABLATION 6: Force all inference to remote (naive baseline)
INFERENCE_ROUTING = {
    "S0": "remote",
    "S1": "remote",
    "S2": "remote",
    "S3": "remote",
    "S5": "remote",
    "default": "remote",
}

# ABLATION 6: Force offload flag (new config flag)
FORCE_OFFLOAD = True

# ABLATION 6: Fixed tier override (new config flag - always use medium)
FIXED_TIER = "medium"

# Not used since always offloading, but kept for consistency
ALWAYS_OFFLOAD_ON_FAST_MOTION = True

LOCAL_YOLO_TIER = "medium"  # Not used if always offloading

# Pre-warm models at startup to avoid cold-start delays
PREWARM_MODELS = True

# Storage directories for results (separate per ablation)
RESULTS_STORAGE_DIR = "storage/data_results/ablation6"
RESULTS_VIDEO_STORAGE_DIR = "storage/video_results/ablation6"

# Local LLM Configuration (not used)
USE_LOCAL_LLM = False
LOCAL_LLM_BACKEND = "moondream"
LOCAL_LLM_MODEL = str(PROJECT_ROOT / "jetson_worker" / "llm" / "models" / "moondream2")
LOCAL_LLM_USE_4BIT = False
LOCAL_LLM_DEVICE = 'cuda'

