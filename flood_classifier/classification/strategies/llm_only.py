import base64
from flood_classifier.inference.llm_image_classifier import LLMImageClassifier
from flood_classifier.postprocessing.classification_formatter import ClassificationFormatter

from ..strategy import ClassificationStrategy


class LLMOnlyStrategy(ClassificationStrategy):
    """Placeholder strategy that relies on image detections only.

    Intended for future LLM-based classification where sensor data is ignored.
    """

    def __init__(self):
        self.image_classifier = LLMImageClassifier()
        self.formatter = ClassificationFormatter()

    def classify(self, detection_results, message):
        b64_image = message.get("image_data", "")
        try:
            image_bytes = base64.b64decode(b64_image)
        except Exception:
            image_bytes = b""
        image_pred = self.image_classifier.classify_flood(image_bytes)
        return self.formatter.format_output(image_pred)
