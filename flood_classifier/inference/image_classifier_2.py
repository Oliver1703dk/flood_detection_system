import numpy as np

class EnhancedImageClassifier:
    def __init__(self, thresholds=(0.05, 0.2), min_detection_conf=0.05):
        # thresholds are tuned values for classifying flood severity.
        self.threshold_low, self.threshold_high = thresholds
        self.min_detection_conf = min_detection_conf

    def calculate_flood_score(self, detection_data, image_size):
        """
        Calculates a flood score by summing individual detection scores.
        Each detection's score is computed as:
            score = confidence * (bbox_area / image_area)
        The overall flood score is the sum of these scores.
        """
        image_area = image_size[0] * image_size[1]
        scores = []
        for d in detection_data:
            conf = d.get("confidence", 0)
            # Filter out detections with very low confidence.
            if conf < self.min_detection_conf:
                continue

            bbox = d.get("bounding_box", [])
            # Unwrap nested bounding boxes if necessary.
            if bbox and (isinstance(bbox[0], list) or isinstance(bbox[0], tuple)):
                bbox = bbox[0]
            if len(bbox) < 4:
                continue
            bbox_area = bbox[2] * bbox[3]
            area_ratio = bbox_area / image_area if image_area != 0 else 0

            score = conf * area_ratio
            scores.append(score)

        overall_score = sum(scores) if scores else 0
        return overall_score

    def classify_flood(self, detection_data, image_size=(640, 640)):
        """
        Classifies flood severity based on the aggregated flood score.
        
        Returns:
            0 (No Flood), 1 (Some Water), or 2 (Flooded)
        """
        score = self.calculate_flood_score(detection_data, image_size)
        # Debug print to trace score calculation.
        print(f"Calculated flood score: {score}")

        if score < self.threshold_low:
            return 0  # No Flood
        elif score < self.threshold_high:
            return 1  # Some Water
        else:
            return 2  # Flooded
