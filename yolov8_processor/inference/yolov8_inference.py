import os
from ultralytics import YOLO

class YOLOv8Inference:
    def __init__(self, model_path=None):
        """
        Initializes the YOLOv8 model.
        :param model_path: Path to the YOLOv8 model file (default: 'model/best.pt').
        """
        if model_path is None:
            # Default path to the model
            model_path = os.path.join(os.path.dirname(__file__), '../model/best.pt')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Ensure the model is placed correctly.")

        self.model = YOLO(model_path)

    def run_inference(self, image):
        """Runs YOLOv8 inference on the given image."""
        results = self.model(image)
        return results
