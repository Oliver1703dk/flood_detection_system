# config.py

# Choose classification mode: "combined" or "fused"
CLASSIFICATION_MODE = "fused"

# Image size for inference (width, height)
IMAGE_SIZE = (640, 640)
# Have this either 'test' or anything else to use a dummy image
IMAGE_MODE = "test"
# Image name for testing
IMAGE_NAME = "flood1"

# Thresholds for EnhancedImageClassifier (or any classifier that uses thresholds)
ENHANCED_THRESHOLD_LOW = 0.1
ENHANCED_THRESHOLD_HIGH = 0.5

# You can add any other configuration variables you need
# For example, MQTT settings or sensor model paths:
MQTT_BROKER_URL = "mqtt.example.com"
MQTT_BROKER_PORT = 1883
MQTT_TOPIC = "your/topic/here"



