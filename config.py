# config.py

# Choose classification mode: "yolo_sensor" or "llm_only"
CLASSIFICATION_MODE =  "yolo_sensor" # "llm_only"  # "yolo_sensor"

# When True, the YOLO+sensor strategy confirms flood predictions using an
# additional LLM-based image detector.
USE_LLM_CONFIRMATION = False

# Image size for inference (width, height)
IMAGE_SIZE = (640, 640)
# Have this either 'test' or anything else to use a dummy image
# IMAGE_MODE = "test"
# Image name for testing
IMAGE_NAME = "flood3"

# Default model size and number used on startup. These will be overridden
# when IMPORTANCE values are used to dynamically select models.
# The following works: nano, small, xlarge
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

IMAGE_MODE = "test_dataset"
# IMAGE_MODE = "test_image"
# IMAGE_MODE = "MQTT_Final"



# Optional default baselines.  
# Set to None to skip seeding defaults and run image-only if no file exists.
USE_DEFAULT_BASELINE = True
DEFAULT_BASELINE = {
    "pre_dawn":  { "temperature_baseline": 22.0, "humidity_baseline": 50.0, "pressure_baseline": 1015.0 },
    "midday":    { "temperature_baseline": 22.0, "humidity_baseline": 50.0, "pressure_baseline": 1015.0 },
    "evening":   { "temperature_baseline": 22.0, "humidity_baseline": 50.0, "pressure_baseline": 1015.0 },
    "night":     { "temperature_baseline": 22.0, "humidity_baseline": 50.0, "pressure_baseline": 1015.0 },
}

