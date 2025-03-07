import os
import json
from datetime import datetime

class DataResultsSaver:
    """
    Saves sensor data, metadata, and classification results for future use,
    grouping files by observation date in the storage/data_results/ directory.
    """

    def __init__(self, storage_dir="storage/data_results"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def save(self, data, classification_result):
        """
        Saves sensor data, metadata, and the classification result into a JSON file.
        The file is stored in a subfolder named by the observation date (YYYY-MM-DD).
        
        Args:
            data (dict): The original data message containing sensor_data and metadata.
            classification_result (int or str): The flood classification result.
        """
        # Prepare the data to save: only sensor_data, metadata, and classification_result
        result_data = {
            "sensor_data": data.get("sensor_data", {}),
            "metadata": data.get("metadata", {}),
            "classification_result": classification_result
        }
        
        # Extract timestamp and camera_id from metadata
        metadata = result_data.get("metadata", {})
        timestamp = metadata.get("timestamp", datetime.utcnow().isoformat())
        camera_id = metadata.get("camera_id", "unknown")

        # Parse timestamp to get the date for folder naming
        try:
            # Replace "Z" with "+00:00" to handle UTC if necessary.
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            observation_date = dt.strftime("%Y-%m-%d")
        except Exception as e:
            print(f"Error parsing timestamp: {e}")
            observation_date = "unknown_date"

        # Create a subfolder for the observation date
        date_dir = os.path.join(self.storage_dir, observation_date)
        os.makedirs(date_dir, exist_ok=True)

        # Create a safe filename using camera_id and timestamp
        safe_timestamp = timestamp.replace(":", "-").replace(".", "-")
        filename = f"{camera_id}_{safe_timestamp}.json"
        filepath = os.path.join(date_dir, filename)

        # Save the result data to the JSON file
        with open(filepath, "w") as f:
            json.dump(result_data, f, indent=4)
        print(f"Data results saved to {filepath}")
