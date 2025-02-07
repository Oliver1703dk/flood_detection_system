import numpy as np

class SensorClassifier:
    """Handles flood classification using SVM on sensor data."""

    def predict(self, model, sensor_data):
        """Runs sensor data through the classifier and returns a flood level."""
        features = np.array([sensor_data["temperature"], sensor_data["humidity"], sensor_data["pressure"]]).reshape(1, -1)
        return model.predict(features)[0]  # Returns 0 (No Flood), 1 (Some Water), or 2 (Flooded)
