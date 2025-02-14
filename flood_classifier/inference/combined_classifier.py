import numpy as np

class CombinedClassifier:
    """
    Combines raw features from image and sensor data into a flood score,
    then classifies the flood condition based on that score.

    The image component is calculated from the number of detections, average
    confidence, and relative bounding box area. The sensor component is computed
    based on deviations from baseline values for temperature, humidity, and pressure.
    
    Parameters can be tuned to reflect the relative reliability of the image data.
    """

    def __init__(self, image_weight=1.0, sensor_weight=1.0,
                 threshold_low=0.2, threshold_high=0.5,
                 sensor_params=None):
        """
        Initializes the classifier with weightings and thresholds.

        Args:
            image_weight (float): Multiplier for the image-derived score.
            sensor_weight (float): Multiplier for the sensor adjustment.
            threshold_low (float): Lower threshold for flood classification.
            threshold_high (float): Upper threshold for flood classification.
            sensor_params (dict): Expected baselines for sensor data. Example:
                {"temperature_baseline": 25, "humidity_baseline": 50, "pressure_baseline": 1013}
        """
        self.image_weight = image_weight
        self.sensor_weight = sensor_weight
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        # Set default sensor baselines if not provided
        self.sensor_params = sensor_params or {
            "temperature_baseline": 20, 
            "humidity_baseline": 50, 
            "pressure_baseline": 1013
        }

    def calculate_image_score(self, image_data, image_size):
        num_detections = len(image_data)
        if num_detections == 0:
            return 0

        confidences = [d["confidence"] for d in image_data]
        avg_confidence = np.mean(confidences)

        areas = []
        for d in image_data:
            bbox = d.get("bounding_box", [])
            # If the bbox is nested (e.g., [[x, y, w, h]]), flatten it.
            if bbox and (isinstance(bbox[0], list) or isinstance(bbox[0], tuple)):
                bbox = bbox[0]
            if len(bbox) < 4:
                continue  # Skip invalid bounding boxes.
            areas.append(bbox[2] * bbox[3])

        total_area = np.sum(areas)
        image_area = image_size[0] * image_size[1]
        area_ratio = total_area / image_area if image_area != 0 else 0

        return num_detections * avg_confidence * area_ratio


    def calculate_sensor_adjustment(self, sensor_data):
        """
        Computes an adjustment score based on sensor data relative to baselines.

        Args:
            sensor_data (list): [temperature, humidity, pressure]

        Returns:
            float: The sensor adjustment value.
        """
        temperature, humidity, pressure = sensor_data
        temp_baseline = self.sensor_params["temperature_baseline"]
        humidity_baseline = self.sensor_params["humidity_baseline"]
        pressure_baseline = self.sensor_params["pressure_baseline"]

        # Example heuristic:
        # - Higher humidity above baseline indicates higher flood risk.
        # - Lower temperature than baseline might support increased risk.
        # - Lower pressure than baseline (often seen with storms) can also signal risk.
        humidity_score = (humidity - humidity_baseline) / 100.0
        temperature_score = (temp_baseline - temperature) / 50.0  # Adjust factor as needed.
        pressure_score = (pressure_baseline - pressure) / 50.0

        sensor_adjustment = humidity_score + temperature_score + pressure_score
        return sensor_adjustment

    def classify(self, input_data, image_size=(1920, 1080)):
        """
        Classifies flood severity by combining image and sensor data.

        Expected Input Format:
        {
            "feature_vector": {
                "image_data": [
                    {"bounding_box": [x, y, width, height], "confidence": c},
                    ...
                ],
                "sensor_data": [temperature, humidity, pressure]
            }
        }

        Returns:
            int: Flood classification: 0 (No Flood), 1 (Some Water), or 2 (Flooded)
        """
        feature_vector = input_data.get("feature_vector", {})
        image_data = feature_vector.get("image_data", [])
        sensor_data = feature_vector.get("sensor_data", [self.sensor_params["temperature_baseline"],
                                                           self.sensor_params["humidity_baseline"],
                                                           self.sensor_params["pressure_baseline"]])
        
        image_score = self.calculate_image_score(image_data, image_size)
        sensor_adjustment = self.calculate_sensor_adjustment(sensor_data)

        # Combine the two scores
        combined_score = self.image_weight * image_score + self.sensor_weight * sensor_adjustment

        # Classify based on thresholds (tuning required)
        if combined_score < self.threshold_low:
            return 0  # No Flood
        elif combined_score < self.threshold_high:
            return 1  # Some Water
        else:
            return 2  # Flooded
