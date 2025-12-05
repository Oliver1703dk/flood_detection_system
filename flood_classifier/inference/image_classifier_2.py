import numpy as np
import config

class EnhancedImageClassifier:
    def __init__(self, thresholds=(0.05, 0.5), min_detection_conf=0.05):
        # thresholds are tuned values for classifying flood severity.
        self.threshold_low, self.threshold_high = thresholds
        self.min_detection_conf = min_detection_conf

    def calculate_flood_score(self, detection_data, image_size, num_models_available=None):
        """
        Calculates a flood score by summing individual detection scores.
        Each detection's score is computed as:
            score = confidence * (bbox_area / image_area)
        The overall flood score is the sum of these scores.
        
        Applies adaptive normalization to scale scores to 3-model baseline:
        - 1 model: scores multiplied by 3.0, confidence boosted 2.5x, agreement = 3
        - 2 models: scores multiplied by 1.5
        - 3 models: no change (normalization = 1.0)
        - N models: scores multiplied by 3.0/N
        """
        image_area = image_size[0] * image_size[1]
        
        # Determine number of models from highest priority source
        if num_models_available is not None:
            num_models = max(int(num_models_available), 1)
        else:
            # Use config.model_number as authoritative source
            num_models = max(getattr(config, "model_number", 3), 1)
            # Optional: try to verify from detection_data (for debugging)
            all_model_ids = set()
            for d in detection_data:
                model_ids = d.get("model_ids", [])
                if isinstance(model_ids, (list, set)):
                    all_model_ids.update(model_ids)
                elif model_ids:
                    all_model_ids.add(model_ids)
            detected_count = len(all_model_ids) if all_model_ids else 0
            # Log if there's a mismatch (but don't change num_models)
            if detected_count > 0 and detected_count != num_models:
                print(f"Warning: Detected {detected_count} model_id(s) in detections, but using config.model_number={num_models} for normalization")
        
        # Normalization: scale TO 3 models
        model_normalization = 3.0 / num_models
        
        # Debug logging for non-standard model counts
        if num_models != 3:
            print(f"Adaptive normalization: Using config.model_number={num_models} model(s), applying {model_normalization:.2f}x normalization")
        
        scores = []
        for d in detection_data:
            conf = d.get("confidence", 0)
            # Filter out detections with very low confidence.
            if conf < self.min_detection_conf:
                continue

            bbox = d.get("bounding_box", [])
            # Unwrap nested bounding boxes if necessary.
            if bbox and (isinstance(bbox[0], list) or isinstance(bbox[0], tuple)):
                bbox = bbox[0]
            if len(bbox) < 4:
                continue
            bbox_area = bbox[2] * bbox[3]
            area_ratio = bbox_area / image_area if image_area != 0 else 0

            # Get model_ids, handling both missing keys and empty lists properly
            model_ids_list = d.get("model_ids", [])
            if not model_ids_list:
                # If missing or empty, default to 1 (at least one model must have detected this)
                model_agreement = 1
            else:
                # Use actual count of agreeing models
                model_agreement = len(model_ids_list)
            #Defaultconfidencemultiplierandagreementboostfor3-modelbaseline
            
            conf_multiplier = 1.0
            effective_agreement = model_agreement
            
            agreement_boost = 1 + 0.2 * (effective_agreement - 1)  # +20% per extra agreeing model
            score = (conf * conf_multiplier) * area_ratio * agreement_boost * model_normalization

            scores.append(score)

        overall_score = sum(scores) if scores else 0
        return overall_score

    def classify_flood(self, detection_data, image_size=(640, 640)):
        """
        Classifies flood severity based on the aggregated flood score.
        
        Returns:
            0 (No Flood), 1 (Some Water), or 2 (Flooded)
        """
        score = self.calculate_flood_score(detection_data, image_size)
        # Debug print to trace score calculation.
        print(f"Calculated flood score: {score}")

        if score < self.threshold_low:
            return 0  # No Flood
        elif score < self.threshold_high:
            return 1  # Some Water
        else:
            return 2  # Flooded
