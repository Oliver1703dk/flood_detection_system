import os
import cv2
from yolov8_processor.inference.yolov8_inference import YOLOv8Inference
from yolov8_processor.classifier.yolov8_final_classifier import YOLOv8FinalClassifier

class MultiModelInference:
    def __init__(self, model_info):
        """
        Initializes multiple YOLOv8 models.
        :param model_info: A list of tuples [(identifier, model_filename), ...]
        """
        self.models = {}
        for model_id, model_filename in model_info:
            self.models[model_id] = YOLOv8Inference(model_filename=model_filename, identifier=model_id)

    def run_all_inference(self, image):
        """
        Runs inference for all models on the given image, then aggregates their predictions
        using the YOLOv8FinalClassifier.
        Returns a single list of detection results.
        """
        results_dict = {}
        for model_id, inference_model in self.models.items():
            print(f"Running inference for model {model_id}...")
            results = inference_model.run_inference(image)
            print(f"Model {model_id} detected {len(results)} objects.")
            results_dict[model_id] = results
        
        # Aggregate/fuse predictions from all models.
        final_classifier = YOLOv8FinalClassifier()
        aggregated_results = final_classifier.classify(results_dict)
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
