import config
from .multi_model_inference import MultiModelInference

class ModelSwitcher:
    """Manage YOLOv8 models and allow switching at runtime."""

    def __init__(self):
        self.model_size = None
        self.model_number = None
        self.multi_inference = None

    def _load_models(self, model_size: str, model_number: int):
        """(Re)load models for the given configuration."""
        model_info = [
            (str(i), f"{model_size}/best{i}.pt") for i in range(1, model_number + 1)
        ]
        self.multi_inference = MultiModelInference(model_info)
        self.model_size = model_size
        self.model_number = model_number

    def switch_models(self, model_size: str, model_number: int):
        """Switch to the provided model configuration if different."""
        if (
            self.multi_inference is None
            or model_size != self.model_size
            or model_number != self.model_number
        ):
            self._load_models(model_size, model_number)

    def switch_by_context(self, context: str):
        """Switch models using a high level context string."""
        mapping = {
            "low_latency": ("nano", 1),
            "balanced": ("nano", 2),
            "high_accuracy": ("small", 3),
        }
        if context not in mapping:
            raise ValueError(f"Unknown context: {context}")
        size, num = mapping[context]
        self.switch_models(size, num)

    def run_inference(self, image, image_name=config.IMAGE_NAME):
        """Run inference using the currently loaded models."""
        if self.multi_inference is None:
            self.switch_models(config.model_size, config.model_number)
        return self.multi_inference.run_all_inference(image, image_name=image_name)

