class FusionStrategy:
    """Combines predictions from the vision and sensor models."""

    def merge_predictions(self, sensor_pred=None, image_pred=None):
        """
        Merges predictions from both models.
        
        - If both predictions exist, take the highest flood classification.
        - If only one is available, use its prediction.
        """

        # TODO: Make this one combine them and not just take the highest flood level
        if sensor_pred is not None and image_pred is not None:
            return max(sensor_pred, image_pred)  # Use the highest flood level
        elif sensor_pred is not None:
            return sensor_pred
        elif image_pred is not None:
            return image_pred
        else:
            return 0  # Default to No Flood
