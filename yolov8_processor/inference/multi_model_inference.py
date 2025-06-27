import os
import cv2
import config
from yolov8_processor.inference.yolov8_inference import YOLOv8Inference
from yolov8_processor.classifier.yolov8_final_classifier import YOLOv8FinalClassifier

class MultiModelInference:
    def __init__(self, model_info=None):
        """Manage multiple YOLOv8 models and allow dynamic switching."""
        self.models = {}
        if model_info is not None:
            self.load_models(model_info)
        else:
            self.reload_from_config()

    def load_models(self, model_info):
        """Load models from a list of (identifier, filename) tuples."""
        self.models = {}
        for model_id, model_filename in model_info:
            self.models[model_id] = YOLOv8Inference(model_filename=model_filename, identifier=model_id)

    def _determine_model_params(self):
        """Decide model size and count from the IMPORTANCE weights."""
        imp = getattr(config, "IMPORTANCE", {})
        energy = imp.get("energy", 0.0)
        timeliness = imp.get("timeliness", 0.0)
        accuracy = imp.get("accuracy", 1.0 - energy - timeliness)

        # Start from the configured defaults
        default_size = getattr(config, "model_size", "nano")
        default_number = getattr(config, "model_number", 3)

        sizes = ["nano", "small", "medium", "large", "xlarge"]
        size_index = sizes.index(default_size) if default_size in sizes else 0

        # Adjust size mostly based on accuracy. Energy and timeliness only
        # push the size down when very high.
        if accuracy > 0.9:
            size_index = min(size_index + 3, len(sizes) - 1)
        elif accuracy > 0.7:
            size_index = min(size_index + 2, len(sizes) - 1)
        elif accuracy > 0.5:
            size_index = min(size_index + 1, len(sizes) - 1)

        if energy > 0.7 or timeliness > 0.7:
            size_index = max(0, size_index - 1)

        model_size = sizes[size_index]

        # Only lower the number of models when timeliness is extremely important
        model_number = default_number
        if timeliness >= 0.85:
            model_number = 1
        elif timeliness >= 0.7:
            model_number = 2

        return model_size, model_number

    def reload_from_config(self):
        """Reload models according to IMPORTANCE settings in config.py."""
        model_size, model_number = self._determine_model_params()
        model_info = [
            (str(i), f"{model_size}/best{i}.pt")
            for i in range(1, model_number + 1)
        ]
        self.load_models(model_info)

    def update_importance(self, importance):
        """Update IMPORTANCE values and reload models."""
        total = sum(importance.values())
        if abs(total - 1.0) > 1e-3:
            raise ValueError("IMPORTANCE values must sum to 1")
        config.IMPORTANCE = importance
        self.reload_from_config()

    def run_all_inference(self, image, image_name=config.IMAGE_NAME):
        """
        Runs inference for all models on the given image, then aggregates their predictions
        using the YOLOv8FinalClassifier.
        Returns a single list of detection results.
        """
        results_dict = {}
        for model_id, inference_model in self.models.items():
            print(f"Running inference for model {model_id} on image {image_name}...")
            results = inference_model.run_inference(image, image_name=image_name)
            print(f"Model {model_id} detected {len(results)} objects.")
            results_dict[model_id] = results
        
        # Aggregate/fuse predictions from all models.
        final_classifier = YOLOv8FinalClassifier()
        aggregated_results = final_classifier.classify_and_draw(results_dict, image, image_name=image_name)

        # aggregated_results = final_classifier.classify(results_dict)
        return aggregated_results


# Example usage:
if __name__ == "__main__":
    # Provide tuples with (identifier, model_filename).
    model_info = [
        ("1", "model1.pt"),
        ("2", "model2.pt"),
        ("3", "model3.pt"),
        ("4", "model4.pt"),
        ("5", "model5.pt"),
    ]
    multi_inference = MultiModelInference(model_info)
    test_image = cv2.imread("test_images/input.jpg")
    all_results = multi_inference.run_all_inference(test_image)
