from abc import ABC, abstractmethod

class ClassificationStrategy(ABC):
    """Base interface for flood classification strategies."""

    @abstractmethod
    def classify(self, detection_results, message):
        """Return a human readable classification result.

        Parameters:
            detection_results: List of detections produced by the vision system.
            message: Original message payload containing at least sensor_data and metadata.
        Returns:
            str: Classification label such as "No Flood" or "Flooded".
        """
        raise NotImplementedError
