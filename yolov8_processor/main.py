from preprocessing.image_processor import ImageProcessor
from inference.yolov8_inference import YOLOv8Inference
from postprocessing.result_formatter import ResultFormatter

class YOLOv8Processor:
    def __init__(self, model_path="best.pt", target_size=(640, 640), confidence_threshold=0.5):
        self.image_processor = ImageProcessor(target_size)
        self.inference_engine = YOLOv8Inference(model_path)
        self.result_formatter = ResultFormatter(confidence_threshold)

    def process(self, input_data):
        """
        Full pipeline for processing an input dataset:
        - Decode and preprocess the image
        - Run YOLOv8 inference
        - Format and filter the results
        """
        # Preprocess the image
        preprocessed_image = self.image_processor.preprocess(input_data["image_data"])

        # Run inference
        inference_results = self.inference_engine.run_inference(
            preprocessed_image,
            metadata=input_data.get("metadata"),
        )

        # Format results
        formatted_results = self.result_formatter.format_results(inference_results)

        # Add sensor data and metadata
        output = {
            "yolo_results": formatted_results,
            "sensor_data": input_data["sensor_data"],
            "metadata": input_data["metadata"],
        }

        return output

# Example usage
if __name__ == "__main__":
    # Mock input data
    input_data = {
        "image_data": "/9j/4AAQSkZJRgABAQAAAQABAAD...",  # Truncated Base64 image
        "sensor_data": {"temperature": 20.3, "humidity": 53, "pressure": 1032.6},
        "metadata": {
            "timestamp": "2025-01-16T19:22:05.883040",
            "location": "50.8503,4.3517",
            "camera_id": "camera_01"
        }
    }

    processor = YOLOv8Processor(model_path="best.pt")
    output = processor.process(input_data)
    print(output)
