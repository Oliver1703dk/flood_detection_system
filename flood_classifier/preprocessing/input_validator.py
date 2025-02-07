class InputValidator:
    """
    Validates structured feature vector input before processing.
    """

    def validate(self, feature_vector):
        """
        Checks if input data follows expected structure.
        """
        if not isinstance(feature_vector, dict) or "feature_vector" not in feature_vector:
            return False, "Invalid input: Missing 'feature_vector' key."

        feature_vector = feature_vector["feature_vector"]

        # Validate image_data
        if "image_data" in feature_vector:
            if not isinstance(feature_vector["image_data"], list):
                return False, "Invalid 'image_data': Expected a list."
            for detection in feature_vector["image_data"]:
                if not isinstance(detection, dict) or "bounding_box" not in detection or "confidence" not in detection:
                    return False, "Invalid detection format."
                if not isinstance(detection["bounding_box"], list) or len(detection["bounding_box"]) != 4:
                    return False, "Bounding box must be a list of 4 numbers."

        # Validate sensor_data
        if "sensor_data" in feature_vector:
            if not isinstance(feature_vector["sensor_data"], list) or len(feature_vector["sensor_data"]) != 3:
                return False, "Invalid 'sensor_data': Expected a list of 3 values."

        return True, "Validation successful."
