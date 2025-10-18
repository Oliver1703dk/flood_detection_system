from datetime import datetime
from dataclasses import asdict, is_dataclass
import os
import json
import base64
import cv2
import numpy as np
import glob
import copy

# Import modules from your project.
from cluster_data_receiver.receiver.mqtt_receiver import MQTTReceiver
from cluster_data_receiver.validation.data_validator import DataValidator
from cluster_data_receiver.storage.storage_manager import StorageManager
from flood_classifier.postprocessing.data_result_saver import DataResultsSaver
from yolov8_processor.inference.multi_model_inference import MultiModelInference
from yolov8_processor.inference.yolov8_inference import YOLOv8Inference
from yolov8_processor.preprocessing.image_processor import ImageProcessor
from yolov8_processor.postprocessing.result_formatter import ResultFormatter
from flood_classifier.classification.strategies import (
    YoloSensorStrategy,
    LLMOnlyStrategy,
    FSMStrategy,
)
from flood_classifier.fsm.flood_fsm import FrameDecision, decision_to_dict


# Import configuration
import config


def simulate_message():
    """
    Simulates an incoming data message.
    Creates a dummy white image, encodes it to Base64,
    and constructs a sample data dictionary.
    """
    

    if(config.IMAGE_MODE == "test"): 
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
    if config.TEST_MOTION is not None:
        message["metadata"]["motion"] = config.TEST_MOTION
    if getattr(config, "TEST_RESOURCE_CONSTRAINED", False):
        message["metadata"]["resource_constrained"] = bool(config.TEST_RESOURCE_CONSTRAINED)
    return message


def load_test_messages(dataset_dir):
    """
    Yields tuples of (message_dict, ground_truth_label), where
    message_dict matches your real ingestion format:
      {
        "image_data": <base64 JPEG>,
        "sensor_data": {...},
        "metadata": {
          "timestamp": "...Z",
          "location": "...",
          "camera_id": "CAM123"
        }
      }
    """
    labels_dir = os.path.join(dataset_dir, "labels")
    images_dir = os.path.join(dataset_dir, "images")

    for json_path in sorted(glob.glob(os.path.join(labels_dir, "*.json"))):
        sample = json.load(open(json_path))

        # 1) encode the image exactly as simulate_message() does
        img = cv2.imread(os.path.join(images_dir, sample["image"]))
        ok, buf = cv2.imencode(".jpg", img)
        b64 = base64.b64encode(buf).decode("utf-8")

        # 2) build the exact same fields
        message = {
            "image_name": sample["image"],
            "image_data": b64,
            "sensor_data": sample["sensor_data"],
            "metadata": {
                "timestamp": sample.get("timestamp", datetime.utcnow().isoformat()+"Z"),
                "location": sample.get("location", "Dataset"),
                # give each test image a valid, slash-free camera_id
                "camera_id": sample.get("camera_id", os.path.splitext(sample["image"])[0])
            }
        }

        meta = message.get("metadata", {})
        if config.TEST_MOTION is not None and "motion" not in meta:
            meta["motion"] = config.TEST_MOTION
        if getattr(config, "TEST_RESOURCE_CONSTRAINED", False) and "resource_constrained" not in meta:
            meta["resource_constrained"] = bool(config.TEST_RESOURCE_CONSTRAINED)
        message["metadata"] = meta

        yield message, sample["label"]


def main():
    classification_mode = config.CLASSIFICATION_MODE  # "yolo_sensor", "llm_only", or "fsm"
    print(f"Using classification mode: {classification_mode}")

    # Instantiate once before looping
    validator = DataValidator()
    storage_manager = StorageManager()
    saver = DataResultsSaver()
    image_processor = ImageProcessor()
    result_formatter = ResultFormatter()

    os.makedirs("storage/data",        exist_ok=True)
    os.makedirs("storage/data_results", exist_ok=True)

    total = correct = 0
    

    # choose between test_dataset or the old dummy
    repeat_count = max(1, getattr(config, "FSM_REPEAT_COUNT", 1))

    if config.IMAGE_MODE == "test_dataset":
        print(f"Using test dataset from {config.TEST_DATASET_DIR}")
        base_samples = list(load_test_messages(config.TEST_DATASET_DIR))
    else:
        print("Using simulated message")
        base_samples = [(simulate_message(), None)]

    iterator = []
    for message, label in base_samples:
        for _ in range(repeat_count):
            iterator.append((copy.deepcopy(message), label))

    # Loop over every JSON+image in your mini-dataset
    for message, gt_label in iterator:
        total += 1
        print(f"\n--- Sample #{total}", 
              f"(ground_truth={gt_label})" if gt_label else "", "---")

        # 1. Validate & store
        if not validator.validate(message):
            print("❌ Validation failed, skipping.")
            continue
        storage_manager.store(message)
        print("✅ Data validated & stored")

        # 2. Preprocess & run YOLOv8 (only in YOLO + sensor mode)
        dets = []
        if classification_mode == "yolo_sensor":
            try:
                pre = image_processor.preprocess(message["image_data"])
            except Exception as e:
                print("Error in preprocessing:", e)
                continue

            print("")
            # model_info = [
            #     (str(i), f"{config.model_size}/best{i}.pt")
            #     for i in range(1, config.model_number+1)
            # ]
            try:
                agg = MultiModelInference().run_all_inference(
                    pre,
                    image_name=message["image_name"],
                    metadata=message.get("metadata"),
                )
            except Exception as e:
                print("Error during inference:", e)
                agg = None

            dets = result_formatter.format_results(agg) if agg else []
            print("Detections:", dets)
        else:
            print("Skipping YOLOv8 inference (LLM-only / FSM mode).")

        # 3. Classify
        strategy_map = {
            "yolo_sensor": YoloSensorStrategy,
            "llm_only": LLMOnlyStrategy,
            "fsm": FSMStrategy,
        }
        strategy_cls = strategy_map.get(classification_mode)
        if strategy_cls is None:
            print("Invalid classification mode, skipping.")
            continue

        strategy = strategy_cls()
        final_result = strategy.classify(dets, message)
        if isinstance(final_result, FrameDecision):
            printable_result = decision_to_dict(final_result)
        elif is_dataclass(final_result):
            printable_result = asdict(final_result)
        else:
            printable_result = final_result
        print("Final result:", printable_result)

        date_folder = message["metadata"]["timestamp"][:10]
        os.makedirs(os.path.join("storage/data_results", date_folder), exist_ok=True)

        # 4. Save & compare to ground truth
        saver.save(message, printable_result)

        if gt_label is not None:
            pred_is_flood = (final_result != "No Flood")
            gt_is_flood   = (gt_label == "flood")
            if pred_is_flood == gt_is_flood:
                print("✅ Correct")
                correct += 1
            else:
                print("❌ Wrong")

    # After all samples, print overall accuracy
    print(f"\n=== Accuracy: {correct}/{total} = {correct/total:.1%} ===")




if __name__ == "__main__":
    main()
