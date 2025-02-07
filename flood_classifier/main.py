from model.model_loader import ModelLoader
from inference.image_classifier import ImageClassifier
from inference.sensor_classifier import SensorClassifier
from inference.fusion_strategy import FusionStrategy
from postprocessing.classification_formatter import ClassificationFormatter

def classify_flood(detection_data, sensor_data, use_sensor_model=True):
    """
    Runs the flood classifier end-to-end.
    
    - Uses YOLOv8-based classification.
    - If `use_sensor_model=True`, integrates sensor-based SVM classification.
    """

    # Load models
    model_loader = ModelLoader()
    sensor_model = None
    if use_sensor_model:
        sensor_model = model_loader.load_sensor_model()

    # Perform YOLO-based classification
    image_classifier = ImageClassifier()
    image_pred = image_classifier.classify_flood(detection_data)

    # Perform SVM-based classification on sensor data
    sensor_pred = None
    if use_sensor_model:
        sensor_pred = SensorClassifier().predict(sensor_model, sensor_data)

    # Merge predictions
    final_prediction = FusionStrategy().merge_predictions(sensor_pred, image_pred)

    # Format output
    return ClassificationFormatter().format_output(final_prediction)

if __name__ == "__main__":
    # Example input
    detection_data = [
        {"bounding_box": [120, 200, 50, 30], "confidence": 0.87},  # Water Detection 1
        {"bounding_box": [100, 150, 40, 20], "confidence": 0.92},  # Water Detection 2
    ]

    sensor_data = {"temperature": 23, "humidity": 70, "pressure": 1012}

    # Classify flood using YOLOv8 + sensor SVM
    print(classify_flood(detection_data, sensor_data, use_sensor_model=True))

    # Classify flood using YOLOv8 only
    print(classify_flood(detection_data, sensor_data, use_sensor_model=False))
