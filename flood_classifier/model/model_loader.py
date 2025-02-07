import joblib
import os

from flood_classifier.model.dummy_sensor_loader import DummySensorModel

class ModelLoader:
    """Loads sensor-based flood classification SVM model."""

    def __init__(self):
        # Model path
        self.sensor_model_path = os.path.join(os.path.dirname(__file__), "sensor_model.pkl")

    def load_sensor_model(self):
        """Loads the trained sensor-based flood classification model."""
        # TODO: Dummy code 
        return DummySensorModel()
        # TODO: Correct code below 
        # if not os.path.exists(self.sensor_model_path):
        #     raise FileNotFoundError(f"Sensor model file not found: {self.sensor_model_path}")
        # return joblib.load(self.sensor_model_path)
