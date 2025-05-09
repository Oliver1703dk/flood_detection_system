# config.py

# Choose classification mode: "combined" or "fused"
CLASSIFICATION_MODE = "combined"

# Image size for inference (width, height)
IMAGE_SIZE = (640, 640)
# Have this either 'test' or anything else to use a dummy image
# IMAGE_MODE = "test"
# Image name for testing
IMAGE_NAME = "no-flood7"

model_size = "nano"

model_number = 3


# You can add any other configuration variables you need
# For example, MQTT settings or sensor model paths:
MQTT_BROKER_URL = "localhost"
MQTT_BROKER_PORT = 1883
MQTT_TOPIC = "flood/detection"


# Path to your test mini-dataset
TEST_DATASET_DIR = "test_dataset"

IMAGE_MODE = "test_dataset"


# Optional default baselines.  
# Set to None to skip seeding defaults and run image-only if no file exists.
USE_DEFAULT_BASELINE = True
DEFAULT_BASELINE = {
    "pre_dawn":  { "temperature_baseline": 22.0, "humidity_baseline": 50.0, "pressure_baseline": 1015.0 },
    "midday":    { "temperature_baseline": 22.0, "humidity_baseline": 50.0, "pressure_baseline": 1015.0 },
    "evening":   { "temperature_baseline": 22.0, "humidity_baseline": 50.0, "pressure_baseline": 1015.0 },
    "night":     { "temperature_baseline": 22.0, "humidity_baseline": 50.0, "pressure_baseline": 1015.0 },
}

