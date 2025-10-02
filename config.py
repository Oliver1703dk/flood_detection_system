# config.py

# Choose classification mode: "yolo_sensor", "llm_only", or "fsm"
CLASSIFICATION_MODE =  "fsm"  # "fsm" or "llm_only"

# When True, the YOLO+sensor strategy confirms flood predictions using an
# additional LLM-based image detector.
USE_LLM_CONFIRMATION = True

# Image size for inference (width, height)
IMAGE_SIZE = (640, 640)
# Have this either 'test' or anything else to use a dummy image
# IMAGE_MODE = "test"
# Image name for testing
IMAGE_NAME = "flood-test-ub3"

# Default model size and number used on startup. These will be overridden
# when IMPORTANCE values are used to dynamically select models.
# Supported sizes: nano, small, medium, large, xlarge
model_size = "small"
model_number = 3

# Importance weights used for dynamic model selection. They must sum to 1.
IMPORTANCE = {
    "energy": 0.33,
    "timeliness": 0.33,
    "accuracy": 0.34,
}


# You can add any other configuration variables you need
# For example, MQTT settings or sensor model paths:
MQTT_BROKER_URL = "localhost"
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
    "pre_dawn":  { "temperature_baseline": 22.0, "humidity_baseline": 50.0, "pressure_baseline": 1015.0 },
    "midday":    { "temperature_baseline": 22.0, "humidity_baseline": 50.0, "pressure_baseline": 1015.0 },
    "evening":   { "temperature_baseline": 22.0, "humidity_baseline": 50.0, "pressure_baseline": 1015.0 },
    "night":     { "temperature_baseline": 22.0, "humidity_baseline": 50.0, "pressure_baseline": 1015.0 },
}


# Default FSM parameters for latency-aware scheduling
FSM_DEFAULTS = {
    "threshold_low": 0.35,
    "threshold_high": 0.65,
    "frames_high": 3,
    "frames_low": 3,
    "frames_ambiguous": 3,
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
    "S5": "local",
    "default": "remote",
}


# Default YOLO tier to keep on the Pi. Tiers above this automatically target
# the Jetson regardless of FSM state. The string must match ModelTier values.
LOCAL_YOLO_TIER = "nano"
