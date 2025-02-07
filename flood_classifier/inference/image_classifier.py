import numpy as np

class ImageClassifier:
    """Handles flood classification using YOLOv8 water detections."""

    def classify_flood(self, detection_data):
        """
        Determines flood severity based on water detection confidence.

        Input:
        - detection_data: List of detected objects (each with confidence & bbox)

        Output:
        - 0 (No Flood), 1 (Some Water), 2 (Flooded)
        """

        if not detection_data:
            return 0  # No Flood

        num_detections = len(detection_data)
        avg_confidence = np.mean([d["confidence"] for d in detection_data])

        # Define flood severity based on detection count and confidence
        # TODO: Fix this to be accurate
        if num_detections == 1 and avg_confidence < 0.85:
            return 1  # Some Water
        elif num_detections > 1 or avg_confidence >= 0.85:
            return 2  # Flooded
        else:
            return 0  # No Flood
