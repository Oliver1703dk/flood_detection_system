class LabelNormalizer:
    def normalize(self, results):
        """
        Normalizes all class labels in the results to "water".
        Assumes each result has a dictionary 'names' mapping class indices to labels.
        """
        for result in results:
            # Replace every label in the names dictionary with "water".
            result.names = {key: "water" for key in result.names.keys()}
        return results
