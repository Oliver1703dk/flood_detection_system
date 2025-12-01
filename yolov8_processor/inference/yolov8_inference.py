import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import config
from path_utils import sanitize_path_segment
from yolov8_processor.inference.label_normalizer import LabelNormalizer

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

    def run_inference(self, image, image_name=config.IMAGE_NAME, run_id=None, metadata=None):
        """Runs YOLOv8 inference on the given image."""
        _ = image_name  # compatibility; naming handled via run_id
        results = self.model(image)
        # Normalize the labels in the results
        normalizer = LabelNormalizer()
        results = normalizer.normalize(results)
        self.draw_bounding_boxes(image, results, image_name=image_name, run_id=run_id, metadata=metadata)
        return results

    def draw_bounding_boxes(self, image, results, image_name=config.IMAGE_NAME, run_id=None, metadata=None):
        """
        Draws bounding boxes on the image based on YOLOv8 detections.
        Saves the image using the model identifier.
        """
        _ = image_name  # retained for backwards compatibility
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

        # New structure: storage/image-detections/ablationX/run_<timestamp>/
        # The run_id is already in the base path (DETECTION_OUTPUT_DIR), so just use it directly
        output_dir = Path(config.DETECTION_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use run_id from metadata or config, or generate timestamp for filename
        metadata = metadata or {}
        if not isinstance(metadata, dict):
            metadata = dict(metadata)
        run_id = metadata.get("run_id") or getattr(config, 'RUN_ID', None)
        timestamp = run_id or datetime.now().strftime("%Y%m%d-%H%M%S-%f")

        identifier = (self.identifier or "").strip()
        if not identifier:
            identifier = "1"

        filename = f"{timestamp}_detection-{identifier}.jpg"
        save_path = output_dir / filename

        cv2.imwrite(str(save_path), image_8u)
        print(f"Detection results saved to {save_path}")

        return image_8u
