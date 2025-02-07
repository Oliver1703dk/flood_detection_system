class Normalization:
    """
    Provides methods to normalize bounding boxes, confidence scores, and sensor data.
    """
    @staticmethod
    def normalize_bounding_boxes(bounding_boxes, img_width, img_height):
        """Normalize bounding box coordinates and retain confidence scores."""
        return [
            [box[0] / img_width, box[1] / img_height, box[2] / img_width, box[3] / img_height, box[4]]
            for box in bounding_boxes
        ]

    @staticmethod
    def standardize_sensor_data(sensor_data):
        """Standardizes sensor data to range [0,1] based on predefined min/max values."""
        min_max = {
            "temperature": (-30, 50),  
            "humidity": (0, 100),
            "pressure": (900, 1100)
        }
        return [(sensor_data[key] - min_max[key][0]) / (min_max[key][1] - min_max[key][0]) for key in min_max.keys()]
