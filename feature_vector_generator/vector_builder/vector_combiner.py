import numpy as np
from vector_builder.normalization import Normalization

class VectorCombiner:
    """
    Combines object detection outputs (bounding boxes & confidence scores) and sensor data into a standardized feature vector.
    """
    def __init__(self):
        self.sensor_keys = ["temperature", "humidity", "pressure"]

    def combine_features(self, detection_data, sensor_data, img_width, img_height):
        """Standardize detection data (from any model) and combine it with sensor data into a unified feature vector."""

        def process_detections(detections):
            """Normalize bounding boxes and retain confidence scores. If no detections, return [0,0,0,0,0]."""
            if not detections:
                return {
                    "bounding_box": [0,0,0,0],
                    "confidence": 0
                }  # Placeholder for "no detection"
            return [
                {
                    "bounding_box": Normalization.normalize_bounding_boxes([obj["bounding_box"]], img_width, img_height)[0],
                    "confidence": obj["confidence"]
                }
                    for obj in detections
                
            ]

        # Process detection data (from any source)
        structured_detections = process_detections(detection_data)


        # normalized_detections = Normalization.normalize_bounding_boxes(standardized_detections, img_width, img_height)

        # Standardize sensor data
        standardized_sensors = Normalization.standardize_sensor_data(sensor_data)

        # Flatten and concatenate
        feature_vector = {
            "image_data": structured_detections,
            "sensor_data": standardized_sensors
        }
        return feature_vector
