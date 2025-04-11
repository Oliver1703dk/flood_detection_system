import os
import json
from datetime import datetime

class DataResultsSaver:
    """
    Saves sensor data, metadata, and classification results for future use,
    grouping files by the current observation date in the storage/data_results/ directory.
    """

    def __init__(self, storage_dir="storage/data_results"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def save(self, data, classification_result):
        """
        Saves sensor data, metadata, and the classification result into a JSON file.
        The file is stored in a subfolder named with today's date (YYYY-MM-DD).
        
        Args:
            data (dict): The original data message containing sensor_data and metadata.
            classification_result (int or str): The flood classification result.
        """
        # Prepare the data to save.
        result_data = {
            "sensor_data": data.get("sensor_data", {}),
            "metadata": data.get("metadata", {}),
            "classification_result": classification_result
        }
        
        # Use the current UTC time for folder naming and filename.
        now = datetime.utcnow()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")
        
        # Extract camera_id from metadata if available.
        camera_id = result_data.get("metadata", {}).get("camera_id", "unknown")
        
        # Create a subfolder for today's date.
        date_dir = os.path.join(self.storage_dir, date_str)
        os.makedirs(date_dir, exist_ok=True)

        # Create a safe filename using camera_id and the current time.
        filename = f"{camera_id}_{time_str}.json"
        filepath = os.path.join(date_dir, filename)

        # Save the result data to the JSON file.
        with open(filepath, "w") as f:
            json.dump(result_data, f, indent=4)
        print(f"Data results saved to {filepath}")
