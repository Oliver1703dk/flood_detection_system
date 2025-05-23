from datetime import datetime
import os
import json
import base64
import cv2
import numpy as np
import glob

# Import modules from your project.
from cluster_data_receiver.receiver.mqtt_receiver import MQTTReceiver
from cluster_data_receiver.validation.data_validator import DataValidator
from cluster_data_receiver.storage.storage_manager import StorageManager
from flood_classifier.baselinecalculator.baseline_calculator import BaselineCalculator
from flood_classifier.inference.classifier_both import ClassifierBoth
from flood_classifier.postprocessing.data_result_saver import DataResultsSaver
from yolov8_processor.inference.multi_model_inference import MultiModelInference
from yolov8_processor.inference.yolov8_inference import YOLOv8Inference
from yolov8_processor.preprocessing.image_processor import ImageProcessor
from yolov8_processor.postprocessing.result_formatter import ResultFormatter
from flood_classifier.postprocessing.classification_formatter import ClassificationFormatter
from flood_classifier.model.model_loader import ModelLoader
from flood_classifier.inference.fusion_strategy import FusionStrategy
from flood_classifier.inference.image_classifier_2 import EnhancedImageClassifier
from flood_classifier.inference.sensor_classifier import SensorClassifier
# Import the CombinedClassifier for one-step classification.


# Import configuration
import config


def simulate_message():
    """
    Simulates an incoming data message.
    Creates a dummy white image, encodes it to Base64,
    and constructs a sample data dictionary.
    """
    

    if(config.IMAGE_MODE == "test_image"):
        print(f'Using test image {config.IMAGE_NAME}')
        # Construct the test image file path.
        test_image_path = os.path.join("test_images", config.IMAGE_NAME +'.jpg')
        
        # Read the image from disk.
        image = cv2.imread(test_image_path)
        if image is None:
            raise ValueError(f"Failed to load test image from path: {test_image_path}")
        print('Image loaded successfully')
    else: 
        # Create a dummy white 640x640 image.
        image = np.ones((config.IMAGE_SIZE[1], config.IMAGE_SIZE[0], 3), dtype=np.uint8) * 255


    # Encode the image as JPEG
    success, buffer = cv2.imencode('.jpg', image)
    if not success:
        raise ValueError("Failed to encode dummy image.")
    
    # Convert the image to a Base64 string
    base64_image = base64.b64encode(buffer).decode('utf-8')

    # Construct sample message.
    message = {
        "image_name": config.IMAGE_NAME + ".jpg",
        "image_data": base64_image,
        "sensor_data": {
            "temperature": 25.0,
            "humidity": 55.0,
            "pressure": 1013.25
        },
        "metadata": {
            "timestamp": "2025-02-07T12:00:00Z",
            "location": "Test Location",
            "camera_id": "CAM123"
        }
    }
    return message



def load_test_dataset(dataset_dir):
    """Yields (message_dict, ground_truth_label) for each JSON+image pair."""
    labels_dir = os.path.join(dataset_dir, "labels")
    images_dir = os.path.join(dataset_dir, "images")

    for jpath in sorted(glob.glob(os.path.join(labels_dir, "*.json"))):
        sample = json.load(open(jpath))
        # read & encode the image
        img = cv2.imread(os.path.join(images_dir, sample["image"]))
        if img is None:
            raise ValueError(f"Cannot read {sample['image']}")
        ok, buf = cv2.imencode(".jpg", img)
        if not ok:
            raise ValueError(f"Encode failed for {sample['image']}")
        b64 = base64.b64encode(buf).decode()

        msg = {
            "image_data": b64,
            "sensor_data": sample["sensor_data"],
            "metadata": {
                "timestamp": sample.get("timestamp", datetime.utcnow().isoformat()+"Z"),
                "location": sample.get("location","Dataset"),
                "camera_id": sample.get("camera_id","N/A"),
            }
        }
        yield msg, sample["label"]  # label is "flood" or "no_flood"


def main():
    # -------------------------------
    # Choose Classification Mode:
    # -------------------------------
    # "combined" uses the CombinedClassifier.
    # "fused" uses separate sensor & image classifiers with fusion.
    classification_mode = config.CLASSIFICATION_MODE  # "fused" or "combined"

    # -------------------------------------------
    # 1. Simulate Incoming Data
    # -------------------------------------------
    try:
        sample_message = simulate_message()
        print("\n--- Simulated Message ---")
        # print(json.dumps(sample_message, indent=4))
    except Exception as e:
        print("Error simulating message:", e)
        return
    

    # # -------------------------------------------
    # # 1. Load the entire test mini-dataset
    # # -------------------------------------------
    # total = correct = 0

    # for sample_message, gt_label in load_test_dataset(config.TEST_DATASET_DIR):
    #     total += 1
    #     print(f"\n--- Sample #{total} (ground_truth={gt_label}) ---")


    # -------------------------------------------
    # 2. Validate and Store Data
    # -------------------------------------------
    validator = DataValidator()
    storage_manager = StorageManager()
    if validator.validate(sample_message):
        storage_manager.store(sample_message)
        print("Data validated and stored successfully.")
    else:
        print("Data validation failed. Exiting simulation.")
        return

    # -------------------------------------------
    # 3. Image Preprocessing & YOLOv8 Inference
    # -------------------------------------------
    image_processor = ImageProcessor()
    try:
        preprocessed_image = image_processor.preprocess(sample_message["image_data"])
        print("Image preprocessed for inference.")
    except Exception as e:
        print("Error in image preprocessing:", e)
        return
    
    

    try:
        # inference = YOLOv8Inference()
        # results = inference.run_inference(preprocessed_image)
        # New multi-model call:
        model_size = config.model_size
        # model_info = [
        #     ("1", f"{model_size}/best1.pt"),
        #     ("2", f"{model_size}/best2.pt"),
        #     ("3", f"{model_size}/best3.pt"),
        #     # ("4", f"{model_size}/best4.pt"),
        #     # ("5", f"{model_size}/best5.pt"),
        # ]
        # model_info takes model_number to know how many models to load
        model_info = [
            (str(i), f"{model_size}/best{i}.pt") for i in range(1, config.model_number + 1)
        ]
        

        multi_inference = MultiModelInference(model_info)
        aggregated_results = multi_inference.run_all_inference(preprocessed_image, image_name=sample_message["image_name"])

        print("YOLOv8 inference completed.")
    except Exception as e:
        print("Error during YOLOv8 inference:", e)
        results = None
        aggregated_results = None

    # Format the YOLO detection results.
    result_formatter = ResultFormatter()
    detection_results = result_formatter.format_results(aggregated_results) if aggregated_results else []
    print("\n--- YOLOv8 Detection Results ---")
    print(detection_results)

    # -------------------------------------------
    # 4. Flood Classification
    # -------------------------------------------
    classification_formatter = ClassificationFormatter()

    if classification_mode == "combined":
        # Use the new combined classifier ClassifierBoth.
        classifier_both = ClassifierBoth(
            baseline_calculator=BaselineCalculator(),
            image_classifier=EnhancedImageClassifier()
        )
        # Convert metadata timestamp to datetime.
        timestamp = datetime.fromisoformat(sample_message["metadata"]["timestamp"].replace("Z", "+00:00"))
        combined_result = classifier_both.classify_flood(
            sensor_data=sample_message["sensor_data"],
            detection_data=detection_results,
            image_size=config.IMAGE_SIZE,
            timestamp=timestamp
        )
        print("Combined flood classification result:")
        print(combined_result)
        final_pred = combined_result["final_prediction"]
        final_result = classification_formatter.format_output(final_pred)
    elif classification_mode == "fused":
        # (A) Sensor-based Classification.
        try:
            model_loader = ModelLoader()
            sensor_model = model_loader.load_sensor_model()
            sensor_classifier = SensorClassifier()
            sensor_pred = sensor_classifier.predict(sensor_model, sample_message["sensor_data"])
            print("Sensor-based flood prediction (0: No Flood, 1: Some Water, 2: Flooded):", sensor_pred)
        except Exception as e:
            print("Error during sensor classification:", e)
            sensor_pred = None

        # (B) Image-based Classification using YOLO detections.
        # image_classifier = ImageClassifier()
        image_classifier = EnhancedImageClassifier()
        image_pred = image_classifier.classify_flood(detection_results)
        print("Image-based flood prediction (0: No Flood, 1: Some Water, 2: Flooded):", image_pred)

        # (C) Fuse the Two Predictions.
        fusion = FusionStrategy()
        final_pred = fusion.merge_predictions(sensor_pred, image_pred)
        final_result = classification_formatter.format_output(final_pred)
    else:
        print("Invalid classification mode selected.")
        return
    
    # Save the data and classification results.
    saver = DataResultsSaver()
    saver.save(sample_message, final_result)

    # -------------------------------------------
    # 5. Update the Baseline Using Latest Data
    # -------------------------------------------
    baseline_calculator = BaselineCalculator(
        results_dir="storage/data_results",
        baseline_file="storage/sensor_baselines.json",
        tau=12
    )

    print("\n🔄 Updating sensor baselines...")
    current_baselines = baseline_calculator.update_baselines()

    if current_baselines:
        print("✅ Updated Baselines:", current_baselines)
    else:
        print("❌ Baseline update failed (no stable period found).")

    print("\n--- Final Flood Classification ---")
    print(final_result)

    # # ─── compare to ground truth ───
    # # any non-"No Flood" counts as “flood”
    # pred_is_flood = (final_result != "No Flood")
    # gt_is_flood   = (gt_label == "flood")
    # if pred_is_flood == gt_is_flood:
    #     print("✅ Correct")
    #     correct += 1
    # else:
    #     print("❌ Wrong")


    # print(f"\n=== Overall: {correct}/{total} = {correct/total:.1%} ===")



    # -------------------------------------------
    # 5. (Optional) Start MQTT Receiver
    # -------------------------------------------
    # Uncomment the following lines to start the MQTT receiver.
    #
    # receiver = MQTTReceiver(
    #     broker_url=config.MQTT_BROKER_URL,  # Replace with your MQTT broker URL.
    #     broker_port=config.MQTT_BROKER_PORT,               # Replace with your MQTT broker port if different.
    #     topic=config.MQTT_TOPIC
    # )
    # receiver.start()


if __name__ == "__main__":
    main()
