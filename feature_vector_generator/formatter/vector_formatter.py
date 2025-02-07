import json

class VectorFormatter:
    """
    Serializes feature vectors into a format suitable for classification.
    """
    def __init__(self):
        pass
    
    def serialize(self, feature_vector):
        """Converts feature vector to JSON format."""
        return json.dumps({"feature_vector": feature_vector})
