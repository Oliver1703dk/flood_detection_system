# config.py

# Choose classification mode: "combined" or "fused"
CLASSIFICATION_MODE = "combined"

# Image size for inference (width, height)
IMAGE_SIZE = (640, 640)
# Have this either 'test' or anything else to use a dummy image
IMAGE_MODE = "test"
# Image name for testing
IMAGE_NAME = "road-green4"


# You can add any other configuration variables you need
# For example, MQTT settings or sensor model paths:
MQTT_BROKER_URL = "localhost"
MQTT_BROKER_PORT = 1883
MQTT_TOPIC = "flood/detection"



