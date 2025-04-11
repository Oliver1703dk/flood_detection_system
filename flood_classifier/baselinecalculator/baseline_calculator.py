import os
import json
import math
from datetime import datetime, timedelta

# --- Helper Function to Determine Time Window ---
def get_time_window(timestamp):
    """
    Define diurnal time windows using UTC hours:
      - pre_dawn: 03:00 to 09:00
      - midday:    09:00 to 15:00
      - evening:   15:00 to 21:00
      - night:     21:00 to 03:00
    """
    hour = timestamp.hour
    if 3 <= hour < 9:
        return "pre_dawn"
    elif 9 <= hour < 15:
        return "midday"
    elif 15 <= hour < 21:
        return "evening"
    else:
        return "night"

# --- Baseline Calibration Class ---
class BaselineCalculator:
    """
    Computes and updates sensor baselines using stable sensor data (at least 24 hours of "No Flood")
    for each time window.
    
    - Overwrites `sensor_baselines.json` with the computed baseline values.
    - (Note: It does not save the latest sensor data since another method is handling that.)
    """
    def __init__(self, 
                 results_dir="storage/data_results", 
                 baseline_file="storage/sensor_baselines.json",
                 required_hours=24, 
                 tau=12):
        self.results_dir = results_dir
        self.baseline_file = baseline_file
        self.required_hours = required_hours
        self.tau = tau  # Decay factor for weighting

    def update_baselines(self):
        """
        Scans stored sensor readings, groups them by time window, computes weighted averages for each window,
        and saves the resulting baseline to file.
        """
        now = datetime.utcnow()
        stable_entries = []

        if not os.path.exists(self.results_dir):
            print("❌ Results directory does not exist.")
            return None

        # Iterate through subfolders (each day) to gather sensor data
        for subfolder in sorted(os.listdir(self.results_dir)):
            subfolder_path = os.path.join(self.results_dir, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            for filename in sorted(os.listdir(subfolder_path)):
                if filename.endswith(".json"):
                    file_path = os.path.join(subfolder_path, filename)
                    try:
                        with open(file_path, "r") as f:
                            data = json.load(f)
                        # Only consider entries labeled as "No Flood"
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

        # Sort entries chronologically and select a stable period
        stable_entries.sort(key=lambda x: x["timestamp"])
        latest_time = stable_entries[-1]["timestamp"]

        stable_period = [stable_entries[-1]]
        max_gap = timedelta(hours=5)
        for entry in reversed(stable_entries[:-1]):
            if (stable_period[0]["timestamp"] - entry["timestamp"]) <= max_gap:
                stable_period.insert(0, entry)
            else:
                break

        if (latest_time - stable_period[0]["timestamp"]).total_seconds() < self.required_hours * 3600:
            print("⚠️ Stable period is too short. Using existing baseline values.")
            return self.get_baselines()

        # Initialize groups for the four time windows
        baselines_by_window = {}
        for window in ["pre_dawn", "midday", "evening", "night"]:
            baselines_by_window[window] = {
                "weighted_sum": {"temperature": 0.0, "humidity": 0.0, "pressure": 0.0},
                "total_weight": {"temperature": 0.0, "humidity": 0.0, "pressure": 0.0},
                "count": 0
            }

        # Group sensor readings and compute weighted sums using exponential decay
        for entry in stable_period:
            time_window = get_time_window(entry["timestamp"])
            baselines_by_window[time_window]["count"] += 1
            delta_hours = (latest_time - entry["timestamp"]).total_seconds() / 3600.0
            weight = math.exp(-delta_hours / self.tau)
            sensor_data = entry.get("sensor_data", {})
            for key in ["temperature", "humidity", "pressure"]:
                value = sensor_data.get(key)
                if value is not None:
                    baselines_by_window[time_window]["weighted_sum"][key] += value * weight
                    baselines_by_window[time_window]["total_weight"][key] += weight

        # Calculate the final baseline for each time window
        baseline = {}
        for window, data in baselines_by_window.items():
            if data["count"] == 0:
                continue
            baseline[window] = {}
            for key in ["temperature", "humidity", "pressure"]:
                if data["total_weight"][key] > 0:
                    baseline[window][key + "_baseline"] = data["weighted_sum"][key] / data["total_weight"][key]
                else:
                    baseline[window][key + "_baseline"] = None

        print("✅ Calculated baselines for each time window:", baseline)

        # Save the new baseline to the baseline file only
        self._write_baseline_file(baseline)
        return baseline

    def _write_baseline_file(self, new_baseline):
        os.makedirs(os.path.dirname(self.baseline_file), exist_ok=True)
        with open(self.baseline_file, "w") as f:
            json.dump(new_baseline, f, indent=4)
        print(f"✅ Baseline file updated: {self.baseline_file}")

    def _write_latest_no_flood_data_to_results(self, stable_entries):
        # This method is no longer used for saving data results.
        pass

    def get_baselines(self):
        if os.path.exists(self.baseline_file):
            with open(self.baseline_file, "r") as f:
                return json.load(f)
        return None

    def get_baseline_for_time(self, timestamp):
        """
        Returns the baseline for the corresponding time window of the given timestamp.
        """
        baselines = self.get_baselines()
        if not baselines:
            return None
        time_window = get_time_window(timestamp)
        return baselines.get(time_window, None)

# --- Example Usage ---
if __name__ == "__main__":
    # Create a BaselineCalculator instance and update baselines
    baseline_calculator = BaselineCalculator()
    baseline = baseline_calculator.update_baselines()

    # Retrieve baseline for a specific time (current UTC time in this example)
    current_time = datetime.utcnow()
    current_baseline = baseline_calculator.get_baseline_for_time(current_time)
    print("Baseline for current time window:", current_baseline)
