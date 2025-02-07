import json
from vector_builder.vector_combiner import VectorCombiner
from validation.input_validator import InputValidator
from formatter.vector_formatter import VectorFormatter

def main():
    """Main function to execute feature vector generation."""

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

    # Step 1: Validate input data
    is_valid, message = validator.validate_inputs(detection_data, sensor_data)
    if not is_valid:
        print(f"Validation Failed: {message}")
        return

    print("Validation Passed.")

    # Step 2: Generate the feature vector
    feature_vector = vector_combiner.combine_features(detection_data, sensor_data, img_width, img_height)

    # Step 3: Serialize the feature vector
    serialized_vector = formatter.serialize(feature_vector)

    print("Feature Vector (Serialized):", serialized_vector)

if __name__ == "__main__":
    main()
