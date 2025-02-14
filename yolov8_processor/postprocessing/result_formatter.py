class ResultFormatter:
    def __init__(self, confidence_threshold=0.30):
        self.confidence_threshold = confidence_threshold

    def format_results(self, results):
        formatted_results = []
        for result in results:
            # Get the dictionary mapping class indices to label names.
            names = result.names  
            for box in result.boxes:
                # Convert confidence to a Python float.
                conf = box.conf.item() if hasattr(box.conf, "item") else box.conf
                # Convert class index (tensor) to integer.
                cls_idx = int(box.cls.item()) if hasattr(box.cls, "item") else int(box.cls)
                # Lookup the label from the names dictionary.
                label = names.get(cls_idx, str(cls_idx))
                if conf >= self.confidence_threshold:
                    formatted_results.append({
                        "label": label,
                        "confidence": conf,
                        "bounding_box": box.xywh.tolist(),
                    })
        return formatted_results
