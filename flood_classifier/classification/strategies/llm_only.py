import config
from flood_classifier.inference.image_classifier_2 import EnhancedImageClassifier
from flood_classifier.postprocessing.classification_formatter import ClassificationFormatter

from ..strategy import ClassificationStrategy


class LLMOnlyStrategy(ClassificationStrategy):
    """Placeholder strategy that relies on image detections only.

    Intended for future LLM-based classification where sensor data is ignored.
    """

    def __init__(self):
        self.image_classifier = EnhancedImageClassifier()
        self.formatter = ClassificationFormatter()

    def classify(self, detection_results, message):
        image_pred = self.image_classifier.classify_flood(
            detection_results,
            image_size=config.IMAGE_SIZE,
        )
        return self.formatter.format_output(image_pred)
