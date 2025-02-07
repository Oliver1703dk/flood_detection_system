import unittest
from ..model.model_loader import ModelLoader

class TestModelLoader(unittest.TestCase):
    def test_load_model(self):
        model_loader = ModelLoader()
        model = model_loader.load_model()
        self.assertIsNotNone(model)  # Ensure model loads properly

if __name__ == '__main__':
    unittest.main()
