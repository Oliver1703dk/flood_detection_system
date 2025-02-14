class FusionStrategy:
    """
    Combines predictions from the vision and sensor models using heuristic adjustments.

    Assumptions:
    - Predictions are integers representing flood severity:
      0 = No Flood, 1 = Some Water, 2 = Flooded.
    - Image data is considered more precise.
    - Sensor data can significantly modify the image prediction if it indicates a stronger or
      weaker flood condition.

    Fusion Logic:
    - If both predictions are available:
         * If sensor_pred is at least one level higher than image_pred, boost the final prediction by one level.
         * If sensor_pred is at least one level lower than image_pred, lower the final prediction by one level.
         * Otherwise, use a weighted average that favors image_pred.
    - If only one prediction is available, return that prediction.
    - If neither is available, default to 0 (No Flood).
    """

    def __init__(self, weight_image=0.8, weight_sensor=0.2):
        self.weight_image = weight_image
        self.weight_sensor = weight_sensor

    def merge_predictions(self, sensor_pred=None, image_pred=None):
        if sensor_pred is not None and image_pred is not None:
            diff = sensor_pred - image_pred
            # If sensor data strongly indicates a higher flood level, boost the prediction
            if diff >= 1:
                return min(image_pred + 1, 2)
            # If sensor data strongly indicates a lower flood level, lower the prediction
            elif diff <= -1:
                return max(image_pred - 1, 0)
            else:
                # If predictions are close, combine them with more weight on the image data.
                combined_score = self.weight_image * image_pred + self.weight_sensor * sensor_pred
                return int(round(combined_score))
        elif sensor_pred is not None:
            return sensor_pred
        elif image_pred is not None:
            return image_pred
        else:
            return 0  # Default to No Flood if both predictions are missing.
