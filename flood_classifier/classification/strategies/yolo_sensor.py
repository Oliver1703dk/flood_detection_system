from datetime import datetime
import base64
import logging

import config
from flood_classifier.baselinecalculator.baseline_calculator import BaselineCalculator
from flood_classifier.inference.classifier_both import ClassifierBoth
from flood_classifier.inference.image_classifier_2 import EnhancedImageClassifier
from flood_classifier.inference.llm_image_classifier import LLMImageDetector
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
        self.llm_detector = LLMImageDetector() if config.USE_LLM_CONFIRMATION else None
        self._logger = logging.getLogger(__name__)

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

        if final_pred in (1, 2) and self.llm_detector is not None:
            b64_image = message.get("image_data", "")
            try:
                image_bytes = base64.b64decode(b64_image)
            except Exception:
                image_bytes = b""
            try:
                llm_pred = self.llm_detector.detect_flood(image_bytes)
                if llm_pred != final_pred:
                    final_pred = llm_pred
            except Exception:
                self._logger.exception("LLM confirmation failed")

        return self.formatter.format_output(final_pred)
