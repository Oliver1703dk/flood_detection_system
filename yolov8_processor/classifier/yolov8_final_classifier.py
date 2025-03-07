
class YOLOv8FinalClassifier:
    def __init__(self, confidence_threshold=0.10):
        self.confidence_threshold = confidence_threshold

    def classify(self, results_dict):
        """
        Aggregates results from multiple YOLOv8 models.
        :param results_dict: Dictionary mapping model_id to YOLOv8 results.
        :return: A single list containing all detection results.
        """
        aggregated_results = []
        for model_id, results in results_dict.items():
            # You could add extra fusion logic here (e.g., majority vote or filtering)
            aggregated_results.extend(results)
        return aggregated_results
    

