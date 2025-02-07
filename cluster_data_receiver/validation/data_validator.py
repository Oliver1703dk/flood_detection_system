class DataValidator:
    """
    Validates incoming data for completeness and correctness.
    """

    def __init__(self):
        # Define required fields and their expected types
        # TODO: Required fields 
        self.required_fields = {
            "image_data": str,
            "sensor_data": dict,
            "metadata": dict
        }

        # TODO: Required fields 
        self.required_sensor_fields = {
            "temperature": (int, float),
            "humidity": (int, float),
            "pressure": (int, float)
        }
        
        # TODO: Required fields 
        self.required_metadata_fields = {
            "timestamp": str,
            "location": str,
            "camera_id": str
        }

    def validate(self, data):
        """
        Validate that the data is complete and adheres to the expected format.

        Args:
            data (dict): The data to validate.

        Returns:
            bool: True if valid, False otherwise.
        """
        # Check top-level fields
        for field, field_type in self.required_fields.items():
            if field not in data:
                print(f"Validation error: Missing field '{field}'")
                return False
            if not isinstance(data[field], field_type):
                print(f"Validation error: Field '{field}' is not of type {field_type.__name__}")
                return False

        # Validate nested fields
        if not self.validate_sensor_data(data["sensor_data"]):
            print("Validation error: Sensor data is invalid")
            return False
        if not self.validate_metadata(data["metadata"]):
            print("Validation error: Metadata is invalid")
            return False

        return True

    def validate_sensor_data(self, sensor_data):
        for field, field_types in self.required_sensor_fields.items():
            if field not in sensor_data:
                print(f"Validation error: Missing sensor field '{field}'")
                return False
            if not isinstance(sensor_data[field], field_types):
                print(f"Validation error: Sensor field '{field}' must be of type {field_types}")
                return False
        return True

    def validate_metadata(self, metadata):
        for field, field_type in self.required_metadata_fields.items():
            if field not in metadata:
                print(f"Validation error: Missing metadata field '{field}'")
                return False
            if not isinstance(metadata[field], field_type):
                print(f"Validation error: Metadata field '{field}' must be of type {field_type.__name__}")
                return False
        return True
