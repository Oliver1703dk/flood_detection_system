import numpy as np

class FeatureExtractor:
    """
    Extracts numerical features from structured input for classification.
    """

    def extract_features(self, feature_vector):
        """
        Converts image detections & sensor data into numerical arrays.

        Expected Input Format:
        {
            "feature_vector": {
                "image_data": [
                    {"bounding_box": [x, y, width, height], "confidence": c},
                    ...
                ],
                "sensor_data": [temperature, humidity, pressure]
            }
        }
        """

        feature_vector = feature_vector.get("feature_vector", {})

        # Extract image detections
        detections = feature_vector.get("image_data", [])
        num_detections = len(detections)
        avg_confidence = np.mean([d["confidence"] for d in detections]) if detections else 0
        avg_bbox_size = np.mean([d["bounding_box"][2] * d["bounding_box"][3] for d in detections]) if detections else 0

        # Extract sensor features
        sensor_features = feature_vector.get("sensor_data", [0, 0, 0])  # Default if missing

        # Return fixed-size numerical feature vector
        return np.array([num_detections, avg_confidence, avg_bbox_size] + sensor_features)
