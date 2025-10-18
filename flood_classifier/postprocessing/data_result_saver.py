import os
import json
from datetime import datetime
from dataclasses import asdict, is_dataclass
from enum import Enum

from path_utils import sanitize_path_segment

def _serialise_result(value):
    """Return a JSON-serialisable representation of ``value``."""
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _serialise_result(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialise_result(v) for v in value]
    return value


class DataResultsSaver:
    """
    Saves sensor data, metadata, and classification results for future use,
    grouping files by the current observation date in the storage/data_results/ directory.
    """

    def __init__(self, storage_dir="storage/data_results", video_storage_dir="storage/video_results"):
        self.storage_dir = storage_dir
        self.video_storage_dir = video_storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(self.video_storage_dir, exist_ok=True)

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
            "classification_result": _serialise_result(classification_result)
        }
        timing_info = data.get("timing")
        if timing_info:
            result_data["timing"] = _serialise_result(timing_info)
        
        # Use the current UTC time for folder naming and filename.
        now = datetime.utcnow()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")

        # Extract camera_id from metadata if available.
        camera_id = result_data.get("metadata", {}).get("camera_id", "unknown")
        video_file = result_data.get("metadata", {}).get("video_file")

        if video_file:
            safe_video_dir = sanitize_path_segment(video_file, fallback="unknown_video")
            target_dir = os.path.join(self.video_storage_dir, safe_video_dir)
        else:
            # Create a subfolder for today's date.
            target_dir = os.path.join(self.storage_dir, date_str)

        os.makedirs(target_dir, exist_ok=True)

        # Create a safe filename using camera_id and the current time.
        filename = f"{sanitize_path_segment(camera_id, fallback='unknown_camera')}_{time_str}.json"
        filepath = os.path.join(target_dir, filename)

        # Save the result data to the JSON file.
        with open(filepath, "w") as f:
            json.dump(result_data, f, indent=4)
        print(f"Data results saved to {filepath}")
        return filepath
