import os
import cv2
import numpy as np
from ultralytics import YOLO
import config

class YOLOv8Inference:
    def __init__(self, model_filename, identifier=""):
        """
        Initializes the YOLOv8 model.
        :param model_filename: The model file name (e.g., "model1.pt").
        :param identifier: A string identifier for the model (e.g., "1").
        """
        # Construct the full model path using a fixed base directory.
        base_dir = os.path.join(os.path.dirname(__file__), "../model")
        model_path = os.path.join(base_dir, model_filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Ensure the model is placed correctly.")

        self.model = YOLO(model_path)
        self.identifier = str(identifier)

    def run_inference(self, image):
        """Runs YOLOv8 inference on the given image."""
        results = self.model(image)
        self.draw_bounding_boxes(image, results)
        return results

    def draw_bounding_boxes(self, image, results):
        """
        Draws bounding boxes on the image based on YOLOv8 detections.
        Saves the image using the model identifier.
        """
        # Convert float images ([0,1]) to 8-bit ([0,255]) if needed.
        if image.dtype in [np.float32, np.float64] and image.max() <= 1.0:
            image_8u = (image * 255).astype(np.uint8)
        else:
            image_8u = image.copy()

        for result in results:
            for box in result.boxes:
                x, y, w, h = map(int, box.xywh.tolist()[0])
                x1, y1 = x - w // 2, y - h // 2
                x2, y2 = x + w // 2, y + h // 2

                conf = box.conf.item() if hasattr(box.conf, "item") else box.conf
                cls_idx = int(box.cls.item()) if hasattr(box.cls, "item") else int(box.cls)
                label = result.names.get(cls_idx, str(cls_idx))

                # Draw the rectangle and label.
                cv2.rectangle(image_8u, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_text = f"{label} ({conf:.2f})"
                cv2.putText(image_8u, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Build the output filename with the identifier.
        filename = f"{config.IMAGE_NAME}"
        if self.identifier:
            filename += f"_{self.identifier}"
        filename += "_output.jpg"
        save_path = os.path.join("test_images/results", filename)
        cv2.imwrite(save_path, image_8u)
        print(f"Detection results saved to {save_path}")

        return image_8u
