from datetime import datetime
from dataclasses import asdict, is_dataclass
import os
import json
import base64
import copy
import cv2
import numpy as np
import glob

# Import modules from your project.
from cluster_data_receiver.receiver.mqtt_receiver import MQTTReceiver
from cluster_data_receiver.validation.data_validator import DataValidator
from cluster_data_receiver.storage.storage_manager import StorageManager
from flood_classifier.baselinecalculator.baseline_calculator import BaselineCalculator
from flood_classifier.postprocessing.data_result_saver import DataResultsSaver
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
    if config.TEST_MOTION is not None:
        message["metadata"]["motion"] = config.TEST_MOTION
    if getattr(config, "TEST_RESOURCE_CONSTRAINED", False):
        message["metadata"]["resource_constrained"] = bool(config.TEST_RESOURCE_CONSTRAINED)
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
        meta = msg.get("metadata", {})
        if config.TEST_MOTION is not None and "motion" not in meta:
            meta["motion"] = config.TEST_MOTION
        if getattr(config, "TEST_RESOURCE_CONSTRAINED", False) and "resource_constrained" not in meta:
            meta["resource_constrained"] = bool(config.TEST_RESOURCE_CONSTRAINED)
        msg["metadata"] = meta
        yield msg, sample["label"]  # label is "flood" or "no_flood"


def main():
    # -------------------------------
    # Choose Classification Mode:
    # -------------------------------
    # "yolo_sensor" uses YOLO detections combined with sensor data.
    # "llm_only" relies on image detections only (placeholder for LLM).
    classification_mode = config.CLASSIFICATION_MODE
    repeat_count = max(1, getattr(config, "FSM_REPEAT_COUNT", 1)) if classification_mode == "fsm" else 1

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
    # 2. Prepare common helpers
    # -------------------------------------------
    validator = DataValidator()
    storage_manager = StorageManager()
    saver = DataResultsSaver()
    baseline_calculator = BaselineCalculator(
        results_dir="storage/data_results",
        baseline_file="storage/sensor_baselines.json",
        tau=12
    )

    # Preload YOLO utilities if needed.
    if classification_mode == "yolo_sensor":
        from yolov8_processor.preprocessing.image_processor import ImageProcessor
        from yolov8_processor.inference.multi_model_inference import MultiModelInference
        from yolov8_processor.postprocessing.result_formatter import ResultFormatter

        image_processor = ImageProcessor()
        model_size = config.model_size
        model_info = [
            (str(i), f"{model_size}/best{i}.pt")
            for i in range(1, config.model_number + 1)
        ]
        multi_inference = MultiModelInference(model_info)
        result_formatter = ResultFormatter()
    else:
        image_processor = None
        multi_inference = None
        result_formatter = None

    # -------------------------------------------
    # 3. Flood Classification Strategy
    # -------------------------------------------
    strategy_map = {
        "yolo_sensor": YoloSensorStrategy,
        "llm_only": LLMOnlyStrategy,
        "fsm": FSMStrategy,
    }
    strategy_cls = strategy_map.get(classification_mode)
    if strategy_cls is None:
        print("Invalid classification mode selected.")
        return

    strategy = strategy_cls()

    for iteration in range(1, repeat_count + 1):
        print(f"\n=== Iteration {iteration}/{repeat_count} ===")
        current_message = copy.deepcopy(sample_message)
        current_metadata = current_message.setdefault("metadata", {})
        current_metadata.setdefault("fsm_iteration", iteration)

        # -------------------------------------------
        # Validate & Store
        # -------------------------------------------
        if not validator.validate(current_message):
            print("❌ Validation failed, skipping iteration.")
            continue
        storage_manager.store(current_message)
        print("✅ Data validated & stored")

        # -------------------------------------------
        # Optional YOLO preprocessing (yolo_sensor mode only)
        # -------------------------------------------
        detection_results = []
        if classification_mode == "yolo_sensor" and image_processor and multi_inference and result_formatter:
            try:
                preprocessed_image = image_processor.preprocess(current_message["image_data"])
                print("Image preprocessed for inference.")
            except Exception as e:
                print("Error in image preprocessing:", e)
                continue

            try:
                aggregated_results = multi_inference.run_all_inference(
                    preprocessed_image,
                    image_name=current_message.get("image_name", config.IMAGE_NAME),
                    metadata=current_message.get("metadata"),
                )
                print("YOLOv8 inference completed.")
            except Exception as e:
                print("Error during YOLOv8 inference:", e)
                aggregated_results = None

            detection_results = (
                result_formatter.format_results(aggregated_results)
                if aggregated_results
                else []
            )
            print("\n--- YOLOv8 Detection Results ---")
            print(detection_results)
        elif classification_mode != "yolo_sensor":
            print("Skipping YOLOv8 inference (LLM-only / FSM mode).")

        # -------------------------------------------
        # Classify
        # -------------------------------------------
        final_result = strategy.classify(detection_results, current_message)

        if isinstance(final_result, FrameDecision):
            printable_result = decision_to_dict(final_result)
        elif is_dataclass(final_result):
            printable_result = asdict(final_result)
        else:
            printable_result = final_result

        saver.save(current_message, printable_result)

        print("\n🔄 Updating sensor baselines...")
        current_baselines = baseline_calculator.update_baselines()

        if current_baselines:
            print("✅ Updated Baselines:", current_baselines)
        else:
            print("❌ Baseline update failed (no stable period found).")

        print("\n--- Flood Classification Result ---")
        print(printable_result)

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
