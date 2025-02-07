import os
import json
from datetime import datetime

class StorageManager:
    """
    Temporarily stores validated data for processing.
    """

    def __init__(self, storage_dir="storage"):
        """
        Initialize the storage manager.

        Args:
            storage_dir (str): Directory where data will be stored.
        """
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def store(self, data):
        """
        Stores validated data in a JSON file.

        Args:
            data (dict): The validated data to store.
        """
        try:
            # Generate a unique filename using timestamp and camera_id
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S%f")
            camera_id = data["metadata"].get("camera_id", "unknown")
            filename = os.path.join(self.storage_dir, f"{camera_id}_{timestamp}.json")

            # Save data to file
            with open(filename, "w") as f:
                json.dump(data, f, indent=4)

            print(f"Data stored successfully at {filename}")
        except Exception as e:
            print(f"Error storing data: {e}")
