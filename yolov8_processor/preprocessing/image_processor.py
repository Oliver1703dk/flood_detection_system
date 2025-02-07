import base64
import cv2
import numpy as np

class ImageProcessor:
    def __init__(self, target_size=(640, 640)):
        self.target_size = target_size

    def decode_image(self, base64_str):
        """Decodes a Base64 image string to a NumPy array."""
        image_data = base64.b64decode(base64_str)
        np_image = np.frombuffer(image_data, np.uint8)
        return cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    def resize_image(self, image):
        """Resizes the image to the target size."""
        return cv2.resize(image, self.target_size)

    def normalize_image(self, image):
        """Normalizes image pixel values to [0, 1]."""
        return image / 255.0

    def preprocess(self, base64_str):
        """Full preprocessing pipeline for an image."""
        image = self.decode_image(base64_str)
        image = self.resize_image(image)
        image = self.normalize_image(image)
        return image
