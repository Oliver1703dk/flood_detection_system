class ClassificationFormatter:
    """Formats the final flood classification result."""

    def format_output(self, prediction):
        labels = {0: "No Flood", 1: "Some Water", 2: "Flooded"}
        return labels.get(prediction, "Unknown Classification")
