import os
import json
import math
from datetime import datetime, timedelta

class BaselineCalculator:
    """
    Computes and updates sensor baselines using stable sensor data (at least 24 hours of "No Flood").
    
    - Overwrites `sensor_baselines.json` each time.
    - Saves the latest gathered sensor values in `storage/data_results/YYYY-MM-DD/HH-MM-SS.json`.
    """

    def __init__(self, 
                 results_dir="storage/data_results", 
                 baseline_file="storage/sensor_baselines.json",
                 required_hours=24, 
                 tau=12):
        self.results_dir = results_dir
        self.baseline_file = baseline_file
        self.required_hours = required_hours
        self.tau = tau  # decay factor for weighting

    def update_baselines(self):
        """
        1. Computes the latest baseline and overwrites `sensor_baselines.json`.
        2. Saves the most recent gathered sensor values into `storage/data_results/YYYY-MM-DD/HH-MM-SS.json`.
        """
        now = datetime.utcnow()
        stable_entries = []

        # Ensure results directory exists
        if not os.path.exists(self.results_dir):
            print("❌ Results directory does not exist.")
            return None

        # Scan ALL folders in data_results/ in chronological order
        subfolders = sorted(os.listdir(self.results_dir))
        for subfolder in subfolders:
            subfolder_path = os.path.join(self.results_dir, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            for filename in sorted(os.listdir(subfolder_path)):
                if filename.endswith(".json"):
                    file_path = os.path.join(subfolder_path, filename)
                    try:
                        with open(file_path, "r") as f:
                            data = json.load(f)
                        if str(data.get("classification_result", "")).strip().lower() != "no flood":
                            continue
                        timestamp_str = data.get("metadata", {}).get("timestamp", None)
                        if not timestamp_str:
                            continue
                        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                        stable_entries.append({
                            "timestamp": timestamp,
                            "sensor_data": data.get("sensor_data", {})
                        })
                    except Exception as e:
                        print(f"❌ Error processing {file_path}: {e}")

        if not stable_entries:
            print("⚠️ No 'No Flood' entries found. Using existing baseline values.")
            return self.get_baselines()

        # Sort entries by timestamp
        stable_entries.sort(key=lambda x: x["timestamp"])

        # Determine the stable period
        latest_time = stable_entries[-1]["timestamp"]
        stable_period = [stable_entries[-1]]
        max_gap = timedelta(hours=5)

        for entry in reversed(stable_entries[:-1]):
            if stable_period and (stable_period[0]["timestamp"] - entry["timestamp"]) <= max_gap:
                stable_period.insert(0, entry)
            else:
                break

        if (latest_time - stable_period[0]["timestamp"]).total_seconds() < self.required_hours * 3600:
            print("⚠️ Stable period is too short. Using latest 'No Flood' sensor readings.")
            return self._write_latest_no_flood_data_to_results(stable_entries)

        # Compute weighted averages
        weighted_sum = {"temperature": 0.0, "humidity": 0.0, "pressure": 0.0}
        total_weight = {"temperature": 0.0, "humidity": 0.0, "pressure": 0.0}

        for entry in stable_period:
            delta_hours = (latest_time - entry["timestamp"]).total_seconds() / 3600.0
            weight = math.exp(-delta_hours / self.tau)
            sensor_data = entry.get("sensor_data", {})
            for key in ["temperature", "humidity", "pressure"]:
                value = sensor_data.get(key)
                if value is not None:
                    weighted_sum[key] += value * weight
                    total_weight[key] += weight

        baseline = {}
        for key in ["temperature", "humidity", "pressure"]:
            baseline[key + "_baseline"] = weighted_sum[key] / total_weight[key] if total_weight[key] > 0 else None

        print("✅ Calculated baseline from stable period:", baseline)

        # Save to `sensor_baselines.json`
        self._write_baseline_file(baseline)

        # Save the latest gathered sensor data to `data_results/YYYY-MM-DD/HH-MM-SS.json`
        self._write_latest_no_flood_data_to_results(stable_entries)

        return baseline

    def _write_baseline_file(self, new_baseline):
        """
        Overwrites the baseline file with the latest computed values.
        """
        os.makedirs(os.path.dirname(self.baseline_file), exist_ok=True)

        # Overwrite with the new baseline
        with open(self.baseline_file, "w") as f:
            json.dump(new_baseline, f, indent=4)

        print(f"✅ Baseline file updated: {self.baseline_file}")

    def _write_latest_no_flood_data_to_results(self, stable_entries):
        """
        Saves the latest gathered sensor values (not the baseline) into the `data_results/YYYY-MM-DD/HH-MM-SS.json` file.
        """
        if not stable_entries:
            print("⚠️ No 'No Flood' sensor data available to save.")
            return

        latest_sensor_data = stable_entries[-1]  # Most recent "No Flood" data
        now = datetime.utcnow()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")

        save_dir = os.path.join(self.results_dir, date_str)
        os.makedirs(save_dir, exist_ok=True)

        file_path = os.path.join(save_dir, f"{time_str}.json")

        # Save the latest sensor reading
        sensor_entry = {
            "sensor_data": latest_sensor_data["sensor_data"],
            "metadata": {
                "timestamp": now.isoformat() + "Z",
                "location": "Latest Sensor Data",
                "camera_id": "LATEST_SENSOR"
            },
            "classification_result": "No Flood"
        }

        with open(file_path, "w") as f:
            json.dump(sensor_entry, f, indent=4)

        print(f"✅ Latest sensor data stored in data_results: {file_path}")

    def get_baselines(self):
        """
        Retrieves the latest baseline values from the baseline file.
        """
        if os.path.exists(self.baseline_file):
            with open(self.baseline_file, "r") as f:
                return json.load(f)
        return None
