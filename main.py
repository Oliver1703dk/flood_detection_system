from feature_vector_generator.vector_builder.vector_combiner import VectorCombiner
from feature_vector_generator.validation.input_validator import InputValidator
from feature_vector_generator.formatter.vector_formatter import VectorFormatter
from flood_classifier.model.model_loader import ModelLoader
from flood_classifier.inference.classifier_inference import ClassifierInference
from flood_classifier.formatter.classification_formatter import ClassificationFormatter

def main():
    """Main function to execute feature vector generation and flood classification."""

    # Example detection data (merged from any model)
    detection_data = [
        {"bounding_box": [120, 200, 50, 30], "confidence": 0.87},  # Detection 1
        {"bounding_box": [100, 150, 40, 20], "confidence": 0.92},  # Detection 2
    ]

    # Example sensor data
    sensor_data = {"temperature": 23, "humidity": 70, "pressure": 1012}

    # Image dimensions (needed for normalization)
    img_width, img_height = 640, 480

    # Initialize components
    validator = InputValidator()
    vector_combiner = VectorCombiner()
    formatter = VectorFormatter()
    model_loader = ModelLoader()
    classifier = ClassifierInference(model_loader)
    classification_formatter = ClassificationFormatter()

    # Step 1: Validate input data
    is_valid, message = validator.validate_inputs(detection_data, sensor_data)
    if not is_valid:
        print(f"Validation Failed: {message}")
        return

    print("Validation Passed.")

    # Step 2: Generate the feature vector
    feature_vector = vector_combiner.combine_features(detection_data, sensor_data, img_width, img_height)

    # Step 3: Classify flood conditions
    classification_result = classifier.classify_flood(feature_vector)

    # Step 4: Format classification result
