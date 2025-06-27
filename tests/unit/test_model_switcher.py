import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from yolov8_processor.inference.model_switcher import ModelSwitcher

class DummyInference:
    def __init__(self, model_info):
        self.model_info = model_info

    def run_all_inference(self, image, image_name=None):
        return self.model_info


def test_switch_models_reuses_instance():
    with patch('yolov8_processor.inference.model_switcher.MultiModelInference', DummyInference):
        switcher = ModelSwitcher()
        switcher.switch_models('nano', 2)
        first = switcher.multi_inference
        assert first.model_info == [('1', 'nano/best1.pt'), ('2', 'nano/best2.pt')]
        # switching with same config should keep instance
        switcher.switch_models('nano', 2)
        assert switcher.multi_inference is first
        # switching with new config loads new instance
        switcher.switch_models('small', 1)
        assert switcher.multi_inference is not first
        assert switcher.model_size == 'small'
        assert switcher.model_number == 1


def test_switch_by_context():
    with patch('yolov8_processor.inference.model_switcher.MultiModelInference', DummyInference):
        switcher = ModelSwitcher()
        switcher.switch_by_context('low_latency')
        assert switcher.model_size == 'nano'
        assert switcher.model_number == 1
        assert isinstance(switcher.multi_inference, DummyInference)

