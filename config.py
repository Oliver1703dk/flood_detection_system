# config.py

from pathlib import Path
from platform_utils import PLATFORM_YOLO_TIER

PROJECT_ROOT = Path(__file__).resolve().parent
DETECTION_OUTPUT_DIR = PROJECT_ROOT / "storage" / "image-detections"

# Choose classification mode: "yolo_sensor", "llm_only", or "fsm"
CLASSIFICATION_MODE =  "fsm"  # "fsm" or "llm_only"

# When True, the YOLO+sensor strategy confirms flood predictions using an
# additional LLM-based image detector.
USE_LLM_CONFIRMATION = False

# When True, the FSM will track energy usage via energy_tracker during
# classification. Disable to skip energy measurement and reduce latency.
ENABLE_FSM_ENERGY_TRACKING = False

# Image size for inference (width, height)
IMAGE_SIZE = (640, 640)
# Have this either 'test' or anything else to use a dummy image
# IMAGE_MODE = "test"
# Image name for testing
IMAGE_NAME = "flood-test-ub3"

# Default model size and number used on startup.
# Supported sizes: nano, small, medium, large, xlarge
model_size = "nano"
model_number = 3



# You can add any other configuration variables you need
# For example, MQTT settings or sensor model paths:
MQTT_BROKER_URL = "192.168.42.10"
MQTT_BROKER_PORT = 1883
MQTT_TOPIC = "sensor/data"


# Path to your test mini-dataset
TEST_DATASET_DIR = "test_dataset"

# IMAGE_MODE = "test_dataset"
IMAGE_MODE = "test_image"
# IMAGE_MODE = "MQTT_Final"

# Default motion hint for simulated/test payloads. Use "fast", "slow", "stop" or None.
TEST_MOTION = "stop"

# When True, mark simulated/test payloads as resource constrained.
TEST_RESOURCE_CONSTRAINED = False




# Optional default baselines.  
# Set to None to skip seeding defaults and run image-only if no file exists.
USE_DEFAULT_BASELINE = True
DEFAULT_BASELINE = {
    "pre_dawn":  { "temperature_baseline": 17.0, "humidity_baseline": 78.0, "pressure_baseline": 1016.0 },
    "midday":    { "temperature_baseline": 17.0, "humidity_baseline": 78.0, "pressure_baseline": 1016.0 },
    "evening":   { "temperature_baseline": 17.0, "humidity_baseline": 78.0, "pressure_baseline": 1016.0 },
    "night":     { "temperature_baseline": 17.0, "humidity_baseline": 78.0, "pressure_baseline": 1016.0 },
}


# Default FSM parameters for latency-aware scheduling
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


# Number of times to replay the same input when running tests in FSM mode.
FSM_REPEAT_COUNT = 5


# MQTT-based remote inference configuration. These values control the
# Pi-to-Jetson request/response loop used to offload heavy models.
MQTT_INFERENCE_REQUEST_TOPIC = "inference/request"
MQTT_INFERENCE_RESPONSE_TOPIC = "inference/response"
MQTT_INFERENCE_HEARTBEAT_TOPIC = "inference/jetson/status"
MQTT_INFERENCE_QOS = 1
MQTT_INFERENCE_TIMEOUT = 6.0  # seconds
MQTT_INFERENCE_MAX_PAYLOAD = 512_000  # bytes before compression


# Backend routing policy keyed by FSM state name. "default" is used when the
# state is not explicitly listed. This keeps routing decisions in config so we
# can fine-tune behaviour without code edits.
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

# Default YOLO tier to keep on the Pi. Tiers above this automatically target
# the Jetson regardless of FSM state. The string must match ModelTier values.
# Platform-specific: nano on Pi, small on Jetson
LOCAL_YOLO_TIER = PLATFORM_YOLO_TIER

# Pre-warm models at startup to avoid cold-start delays
# Set to False to disable model pre-warming (models will load on first use)
PREWARM_MODELS = True



# Local LLM Configuration
USE_LOCAL_LLM = True  # Set to True to use local VLM instead of OpenAI API

# Model options:

# Default local VLM backend. Currently supported: "moondream".
LOCAL_LLM_BACKEND = "moondream"

# Path or model identifier for the selected backend.
LOCAL_LLM_MODEL = str(PROJECT_ROOT / "jetson_worker" / "llm" / "models" / "moondream2")

# Use 4-bit quantization to reduce memory usage (highly recommended for Jetson)
LOCAL_LLM_USE_4BIT = False

# Device for local LLM ('cuda', 'cpu', or None for auto-detect)
LOCAL_LLM_DEVICE = 'cuda'  
