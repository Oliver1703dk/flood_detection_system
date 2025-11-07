import base64
import cv2
import numpy as np
from time import perf_counter
from typing import Dict, Optional

class ImageProcessor:
    def __init__(self, target_size=(640, 640)):
        self.target_size = target_size

    def decode_image(self, base64_str, timing_dict: Optional[Dict[str, float]] = None):
        """Decodes a Base64 image string to a NumPy array."""
        decode_start = perf_counter()
        image_data = base64.b64decode(base64_str)
        decode_duration = perf_counter() - decode_start
        if timing_dict is not None:
            timing_dict["image_decode_s"] = decode_duration
        
        imdecode_start = perf_counter()
        np_image = np.frombuffer(image_data, np.uint8)
        result = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        imdecode_duration = perf_counter() - imdecode_start
        if timing_dict is not None:
            timing_dict["image_imdecode_s"] = imdecode_duration
        return result

    def resize_image(self, image, timing_dict: Optional[Dict[str, float]] = None):
        """Resizes the image to the target size."""
        resize_start = perf_counter()
        result = cv2.resize(image, self.target_size)
        resize_duration = perf_counter() - resize_start
        if timing_dict is not None:
            timing_dict["image_resize_s"] = resize_duration
        return result

    def normalize_image(self, image):
        """Normalizes image pixel values to [0, 1]."""
        return image / 255.0

    def preprocess(self, base64_str, timing_dict: Optional[Dict[str, float]] = None):
        """Full preprocessing pipeline for an image."""
        preprocess_start = perf_counter()
        image = self.decode_image(base64_str, timing_dict=timing_dict)
        image = self.resize_image(image, timing_dict=timing_dict)
        # TODO: Commented this out to test the model without normalization
        # image = self.normalize_image(image)
        preprocess_duration = perf_counter() - preprocess_start
        if timing_dict is not None:
            timing_dict["image_preprocess_total_s"] = preprocess_duration
        return image
