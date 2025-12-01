import os
import sys
from pathlib import Path
import cv2
from datetime import datetime
import time
import numpy as np
import config

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yolov8_processor.inference.yolov8_inference import YOLOv8Inference
from yolov8_processor.classifier.yolov8_final_classifier import YOLOv8FinalClassifier
from flood_classifier.utils.energy_tracker import get_energy_tracker

class MultiModelInference:
    def __init__(self, model_info=None):
        """Manage multiple YOLOv8 models and allow dynamic switching."""
        self.models = {}
        self.last_energy_metrics = {}  # Store energy metrics from last inference
        # if model_info is not None:
        #     self.load_models(model_info)
        # else:
        self.reload_from_config()

    def load_models(self, model_info):
        """Load models from a list of (identifier, filename) tuples."""
        self.models = {}
        for model_id, model_filename in model_info:
            try:
                self.models[model_id] = YOLOv8Inference(
                    model_filename=model_filename, identifier=model_id
                )
            except FileNotFoundError as e:
                # Skip models that are not available and continue loading others.
                print(f"Warning: {e}. Skipping model {model_id}.")

    def _determine_model_params(self):
        """Use model_size and model_number directly from config."""
        model_size = getattr(config, "model_size", "nano")
        model_number = getattr(config, "model_number", 3)
        return model_size, model_number

    def reload_from_config(self):
        """Reload models according to model_size and model_number settings in config.py."""
        model_size, model_number = self._determine_model_params()
        use_baseline = getattr(config, "USE_BASELINE_MODELS", False)
        
        # Use baseline models for single-model ablations when USE_BASELINE_MODELS is True
        if use_baseline and model_number == 1:
            model_info = [
                (str(i), f"baseline/{model_size}/best{i}.pt")
                for i in range(1, model_number + 1)
            ]
        else:
            model_info = [
                (str(i), f"{model_size}/best{i}.pt")
                for i in range(1, model_number + 1)
            ]
        self.load_models(model_info)

    def update_importance(self, importance):
        """Deprecated: IMPORTANCE is no longer used. This method does nothing."""
        # IMPORTANCE-based model selection has been removed.
        # Models are now determined by model_size and model_number in config.
        pass

    def run_all_inference(self, image, image_name=config.IMAGE_NAME, metadata=None):
        """
        Runs inference for all models on the given image, then aggregates their predictions
        using the YOLOv8FinalClassifier.
        Returns a single list of detection results.
        """
        print(image_name)
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        results_dict = {}
        
        # Store run_id in metadata for use in saving paths
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = dict(metadata)
        metadata["run_id"] = run_id
        
        # Track energy for YOLO inference
        energy_tracker = get_energy_tracker()
        energy_metrics = {}
        
        with energy_tracker.measure("yolo_inference") as metrics:
            for model_id, inference_model in self.models.items():
                print(f"Running inference for model {model_id} on image {image_name}...")
                results = inference_model.run_inference(
                    image,
                    image_name=image_name,
                    run_id=run_id,
                    metadata=metadata,
                )
                print(f"Model {model_id} detected {len(results)} objects.")
                results_dict[model_id] = results
            energy_metrics.update(metrics)
        
        # Store energy metrics for later retrieval by FSM
        self.last_energy_metrics = energy_metrics.copy()
        
        # Aggregate/fuse predictions from all models.
        final_classifier = YOLOv8FinalClassifier()
        aggregated_results = final_classifier.classify_and_draw(
            results_dict,
            image,
            image_name=image_name,
            run_id=run_id,
            metadata=metadata,
        )

        # aggregated_results = final_classifier.classify(results_dict)
        return aggregated_results

    def prewarm_inference(self):
        """
        Run dummy inference on loaded models to eliminate cold-start delays.
        Only prewarms nano and small models (Pi-local models).
        Medium and large models run on Jetson and don't need Pi prewarming.
        This initializes CUDA/CPU contexts, allocates memory, and warms up the models.
        """
        if not self.models:
            print("⚠️ No models loaded to prewarm")
            return
        
        # Only prewarm nano and small models (Pi-local models)
        model_size = getattr(config, "model_size", "nano").lower()
        if model_size not in ("nano", "small"):
            print(f"ℹ️ Skipping prewarm for {model_size} models (only nano/small are prewarmed on Pi)")
            return
        
        print(f"🔥 Pre-warming inference on {model_size} models (Pi-local models only)...")
        image_size = getattr(config, "IMAGE_SIZE", (640, 640))
        dummy_image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
        print(f"📐 Created dummy image for warmup: {image_size[0]}x{image_size[1]}")
        
        warmup_start = time.perf_counter()
        for model_id, inference_model in self.models.items():
            try:
                print(f"🔥 Running warmup inference for {model_size} model {model_id}...")
                # Run dummy inference to trigger full initialization
                _ = inference_model.model(dummy_image)  # Direct YOLO call
                print(f"  ✓ Model {model_id} warmup complete")
            except Exception as warmup_exc:
                print(f"⚠️ Warmup failed for model {model_id}: {warmup_exc}")
        
        warmup_time = time.perf_counter() - warmup_start
        print(f"✅ Inference warmup complete for {len(self.models)} {model_size} model(s) in {warmup_time:.3f}s")
        print("🎯 Pi-local models ready - no cold-start delays expected")


# Example usage:
if __name__ == "__main__":
    # Provide tuples with (identifier, model_filename).
    model_info = [
        ("1", "model1.pt"),
        ("2", "model2.pt"),
        ("3", "model3.pt"),
        ("4", "model4.pt"),
        ("5", "model5.pt"),
    ]
    multi_inference = MultiModelInference(model_info)
    test_image = cv2.imread("test_images/input.jpg")
    all_results = multi_inference.run_all_inference(test_image)
