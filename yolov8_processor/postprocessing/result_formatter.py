class ResultFormatter:
    def __init__(self, confidence_threshold=0.30):
        self.confidence_threshold = confidence_threshold

    def format_results(self, results):
        """
        Filters and formats YOLOv8 results.

        Expected that each result has a 'boxes' attribute with objects containing:
        - label
        - confidence
        - xywh: bounding box coordinates (converted to list)
        """
        formatted_results = []
        for result in results:
            for box in result.boxes:
                if box.confidence >= self.confidence_threshold:
                    formatted_results.append({
                        "label": box.label,
                        "confidence": box.confidence,
                        "bounding_box": box.xywh.tolist(),
                    })
        return formatted_results
