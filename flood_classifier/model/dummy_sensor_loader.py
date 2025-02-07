class DummySensorModel:
    def predict(self, X):
        # Regardless of input, always return [0] (No Flood).
        return [0]
