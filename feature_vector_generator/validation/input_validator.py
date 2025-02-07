
class InputValidator:
    """
    Ensures consistency and completeness of object detection outputs and sensor data.
    """
    def __init__(self):
        self.required_sensor_keys = {"temperature", "humidity", "pressure"}
        self.required_bbox_keys = {"bounding_box", "confidence"}
    
    def validate_inputs(self, detection_data, sensor_data):
        """Validate detection outputs and sensor data."""
        if detection_data is None or not isinstance(detection_data, list):
            return False, "Invalid detection data format. Expected a list of detections."
        
        for detection in detection_data:
            if not isinstance(detection, dict):
                return False, "Each detection must be a dictionary."
            if not all(key in detection for key in self.required_bbox_keys):
                return False, "Bounding box data is incomplete."
            if not isinstance(detection["bounding_box"], list) or len(detection["bounding_box"]) != 4:
                return False, "Bounding box must be a list of 4 values (x, y, width, height)."
            if not isinstance(detection["confidence"], (int, float)) or not (0 <= detection["confidence"] <= 1):
                return False, "Confidence must be a float between 0 and 1."
        
        if not isinstance(sensor_data, dict):
            return False, "Invalid sensor data format. Expected a dictionary."
        if not self.required_sensor_keys.issubset(sensor_data.keys()):
            return False, "Missing sensor data fields."
        if not all(isinstance(sensor_data[key], (int, float)) for key in self.required_sensor_keys):
            return False, "All sensor values must be numeric."
        
        print(detection_data)
        print(sensor_data)
        
        return True, "Validation successful."