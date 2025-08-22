from datetime import datetime

import config
from flood_classifier.baselinecalculator.baseline_calculator import BaselineCalculator
from flood_classifier.inference.classifier_both import ClassifierBoth
from flood_classifier.inference.image_classifier_2 import EnhancedImageClassifier
from flood_classifier.postprocessing.classification_formatter import ClassificationFormatter

from ..strategy import ClassificationStrategy


class YoloSensorStrategy(ClassificationStrategy):
    """Classification using YOLO detections combined with sensor data."""

    def __init__(self):
        self.classifier = ClassifierBoth(
            baseline_calculator=BaselineCalculator(),
            image_classifier=EnhancedImageClassifier(),
        )
        self.formatter = ClassificationFormatter()

    def classify(self, detection_results, message):
        timestamp_str = message.get("metadata", {}).get("timestamp")
        timestamp = None
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except Exception:
                timestamp = None

        combined = self.classifier.classify_flood(
            sensor_data=message.get("sensor_data", {}),
            detection_data=detection_results,
            image_size=config.IMAGE_SIZE,
            timestamp=timestamp,
        )
        final_pred = combined["final_prediction"]
        return self.formatter.format_output(final_pred)
