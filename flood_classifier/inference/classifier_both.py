from datetime import datetime
from flood_classifier.baselinecalculator.baseline_calculator import BaselineCalculator
from flood_classifier.inference.image_classifier_2 import EnhancedImageClassifier


class ClassifierBoth:
    """
    Integrates sensor anomaly detection with the image-based flood classifier.
    
    Sensor anomalies (ΔRH, ΔT, ΔP) computed from the baseline are used to boost the YOLO
    flood score. The final combined score is then used for classification.
    """
    
    def __init__(self, 
                 baseline_calculator=None, 
                 image_classifier=None,
                 humidity_threshold=25,       # ΔRH threshold
                 temperature_threshold=-2.5,      # ΔT threshold (i.e. drop > 2°C)
                 pressure_threshold=-5,         # ΔP threshold (i.e. drop > 1 unit)
                 humidity_weight=0.1,           # Weight for humidity anomaly
                 temperature_weight=0.05,       # Weight for temperature anomaly
                 pressure_weight=0.03,          # Weight for pressure anomaly
                 threshold_low=0.3,             # Combined score low threshold
                 threshold_high=0.6):           # Combined score high threshold
        self.baseline_calculator = baseline_calculator or BaselineCalculator()
        self.image_classifier = image_classifier or EnhancedImageClassifier()
        self.humidity_threshold = humidity_threshold
        self.temperature_threshold = temperature_threshold
        self.pressure_threshold = pressure_threshold
        self.humidity_weight = humidity_weight
        self.temperature_weight = temperature_weight
        self.pressure_weight = pressure_weight
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high

    def classify_flood(self, sensor_data, detection_data, image_size=(640, 640), timestamp=None):
        if timestamp is None:
            timestamp = datetime.utcnow()

        # Retrieve baseline for the current time window
        baseline = self.baseline_calculator.get_baseline_for_time(timestamp)
        anomalies = {}
        sensor_boost = 0
        if baseline is not None:
            # Compute sensor deltas for each metric
            for key in ["temperature", "humidity", "pressure"]:
                baseline_val = baseline.get(key + "_baseline")
                current_val = sensor_data.get(key)
                if baseline_val is not None and current_val is not None:
                    anomalies["delta_" + key] = current_val - baseline_val
                else:
                    anomalies["delta_" + key] = 0

            # # Apply thresholds to determine boost to the flood score
            # if anomalies.get("delta_humidity", 0) > self.humidity_threshold:
            #     sensor_boost += self.humidity_weight
            # if anomalies.get("delta_temperature", 0) < self.temperature_threshold:
            #     sensor_boost += self.temperature_weight
            # if anomalies.get("delta_pressure", 0) < self.pressure_threshold:
            #     sensor_boost += self.pressure_weight
            # NEW ---------------------------------------------
            dh = anomalies.get("delta_humidity", 0)
            dt = anomalies.get("delta_temperature", 0)
            dp = anomalies.get("delta_pressure", 0)

            if dh > self.humidity_threshold:
                sensor_boost += self._graduated_weight(
                    dh, self.humidity_threshold, self.humidity_weight)

            if dt < self.temperature_threshold:
                sensor_boost += self._graduated_weight(
                    dt, self.temperature_threshold, self.temperature_weight)

            if dp < self.pressure_threshold:
                sensor_boost += self._graduated_weight(
                    dp, self.pressure_threshold, self.pressure_weight)

        else:
            print("Baseline not available; relying on image classifier only.")

        # Calculate image-based flood score
        image_score = self.image_classifier.calculate_flood_score(detection_data, image_size)
        # Combined score is the sum of the image score and the sensor boost
        combined_score = image_score + sensor_boost

        print(f"Image score: {image_score}")
        print(f"Sensor boost: {sensor_boost}")
        print(f"Combined score: {combined_score}")

        # Determine final prediction based on the combined score
        if combined_score < self.threshold_low:
            final_prediction = 0  # No Flood
        elif combined_score < self.threshold_high:
            final_prediction = 1  # Some Water
        else:
            final_prediction = 2  # Flooded

        return {
            "final_prediction": final_prediction,
            "combined_score": combined_score,
            "image_score": image_score,
            "sensor_boost": sensor_boost,
            "anomalies": anomalies,
            "baseline": baseline
        }


    # --- NEW helper -----------------------------------------
    @staticmethod
    def _graduated_weight(delta, gate, base_weight, severe_factor=2):
        """
        Returns base_weight if delta crosses gate once,
        and (severe_factor * base_weight) if it crosses twice.
        Example: gate = -5  →  severe tier at -10.
        """
        if (gate < 0 and delta < gate * severe_factor) or \
           (gate > 0 and delta > gate * severe_factor):
            return base_weight * severe_factor
        return base_weight

