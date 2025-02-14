import numpy as np

class EnhancedImageClassifier:
    def __init__(self, thresholds=(0.1, 0.5)):
        # thresholds are empirically tuned values: (low threshold, high threshold)
        self.threshold_low, self.threshold_high = thresholds

    def calculate_flood_score(self, detection_data, image_size):
        """
        Calculates a flood score based on image detections.

        Parameters:
        - detection_data: List of detections. Each detection should be a dict with keys:
          "bounding_box": [x, y, width, height] and "confidence": float
        - image_size: Tuple (width, height) representing the dimensions of the image.
        """
        num_detections = len(detection_data)
        if num_detections == 0:
            return 0
        
        confidences = [d["confidence"] for d in detection_data]
        avg_confidence = np.mean(confidences)
        
        # Calculate total bounding box area and relative coverage using "bounding_box"
        areas = [d["bounding_box"][2] * d["bounding_box"][3] for d in detection_data]
        total_area = np.sum(areas)
        image_area = image_size[0] * image_size[1]
        area_ratio = total_area / image_area
        
        # Combine features into a flood score
        score = num_detections * avg_confidence * area_ratio
        return score

    def classify_flood(self, detection_data, image_size=(1920, 1080)):
        """
        Classifies flood severity based on detection data and image size.
        
        Returns:
        - 0 (No Flood), 1 (Some Water), or 2 (Flooded)
        """
        score = self.calculate_flood_score(detection_data, image_size)
        
        # Determine flood level based on tuned thresholds
        if score < self.threshold_low:
            return 0  # No Flood
        elif score < self.threshold_high:
            return 1  # Some Water
        else:
            return 2  # Flooded
