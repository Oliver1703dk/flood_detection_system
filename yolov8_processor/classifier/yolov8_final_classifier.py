import os
import cv2
import numpy as np
import config  # Import config to access IMAGE_NAME

class AggregatedBox:
    def __init__(self, xywh, conf, cls, model_ids=None):
        self.xywh = np.array(xywh)
        self.conf = conf
        self.cls = cls
        self.model_ids = set(model_ids) if model_ids else set()


class AggregatedResult:
    def __init__(self, names, boxes):
        # Mimics YOLOv8 result structure.
        self.names = names  # e.g., {0: "water"}
        self.boxes = boxes  # List of AggregatedBox instances

class YOLOv8FinalClassifier:
    def __init__(self, confidence_threshold=0.03, iou_threshold=0.5):
        """
        :param confidence_threshold: Minimum confidence required for a detection to be considered.
        :param iou_threshold: IoU threshold used to group overlapping detections.
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

    def classify(self, results_dict):
        """
        Aggregates results from multiple YOLOv8 models using confidence-weighted voting
        with spatial grouping.
        
        :param results_dict: Dictionary mapping model_id to YOLOv8 results.
        :return: A list with a single aggregated result mimicking YOLOv8 output.
        """
        # Collect all detections from all models.
        all_detections = []
        for model_id, results in results_dict.items():
            for result in results:
                # Iterate over each detection box.
                for box in result.boxes:
                    conf = box.conf.item() if hasattr(box.conf, "item") else box.conf
                    cls_idx = int(box.cls.item()) if hasattr(box.cls, "item") else int(box.cls)
                    # Convert xywh to a simple list.
                    xywh = box.xywh.tolist()[0]  # Assuming one detection per box list entry.
                    if conf >= self.confidence_threshold:
                        all_detections.append({
                            "xywh": xywh,
                            "conf": conf,
                            "cls": cls_idx,
                            "model_id": model_id
                        })


        # Group overlapping detections and aggregate their confidences.
        aggregated_boxes = self.group_detections(all_detections)

        # Build an aggregated result mimicking YOLOv8 output.
        # Here we assume the only label is "water" (with class index 0).
        aggregated_result = AggregatedResult(names={0: "water"}, boxes=aggregated_boxes)
        return [aggregated_result]

    def group_detections(self, detections):
        """
        Groups detections that are spatially overlapping (based on IoU)
        and aggregates their bounding boxes and confidence scores.
        
        :param detections: List of detections as dicts with keys: 'xywh', 'conf', 'cls'
        :return: List of AggregatedBox instances.
        """
        grouped_boxes = []
        used = [False] * len(detections)
        
        for i in range(len(detections)):
            if used[i]:
                continue
            group = [detections[i]]
            model_ids = {detections[i]['model_id']}
            used[i] = True

            for j in range(i + 1, len(detections)):
                if used[j]:
                    continue
                # Compute the IoU between detection i and detection j.
                iou = self.compute_iou(detections[i]['xywh'], detections[j]['xywh'])
                if iou >= self.iou_threshold:
                    group.append(detections[j])
                    model_ids.add(detections[j]['model_id'])
                    used[j] = True
            # Aggregate group: compute weighted average for bounding box and sum confidences.
            total_conf = sum(d['conf'] for d in group)
            weighted_box = [0, 0, 0, 0]
            for d in group:
                for k in range(4):
                    weighted_box[k] += d['xywh'][k] * d['conf']
            weighted_box = [val / total_conf for val in weighted_box]
            # Use the class from the first detection (they should all be "water")
            aggregated_box = AggregatedBox(weighted_box, total_conf, group[0]['cls'], model_ids=model_ids)

            grouped_boxes.append(aggregated_box)
        
        return grouped_boxes

    def compute_iou(self, xywh1, xywh2):
        """
        Computes Intersection over Union (IoU) between two boxes given in xywh format.
        Here, xywh is in the format: [center_x, center_y, width, height].
        
        :param xywh1: List or array [x, y, w, h]
        :param xywh2: List or array [x, y, w, h]
        :return: IoU value.
        """
        x1, y1, x2, y2 = self.xywh_to_xyxy(xywh1)
        x1_p, y1_p, x2_p, y2_p = self.xywh_to_xyxy(xywh2)

        inter_x1 = max(x1, x1_p)
        inter_y1 = max(y1, y1_p)
        inter_x2 = min(x2, x2_p)
        inter_y2 = min(y2, y2_p)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2_p - x1_p) * (y2_p - y1_p)
        union_area = area1 + area2 - inter_area

        if union_area == 0:
            return 0
        return inter_area / union_area

    def xywh_to_xyxy(self, xywh):
        """
        Converts bounding box format from xywh (center_x, center_y, width, height)
        to xyxy (top-left x, top-left y, bottom-right x, bottom-right y).
        
        :param xywh: List or array [x, y, w, h]
        :return: Tuple (x1, y1, x2, y2)
        """
        x, y, w, h = xywh
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return x1, y1, x2, y2

    def draw_aggregated_bounding_boxes(self, image, aggregated_results, image_name=config.IMAGE_NAME):
        """
        Draws aggregated bounding boxes on the provided image and saves the image.
        The image is saved to the same folder as your YOLOv8Inference output.
        
        Parameters:
            image (numpy.ndarray): The image on which to draw the boxes.
            aggregated_results (list): List of AggregatedResult objects.
        
        Returns:
            numpy.ndarray: The image with drawn bounding boxes.
        """
        for agg_res in aggregated_results:
            for box in agg_res.boxes:
                # box.xywh is a numpy array in the format [center_x, center_y, width, height]
                x, y, w, h = box.xywh
                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)
                
                # Retrieve label (should be "water") and confidence.
                label = agg_res.names.get(box.cls, str(box.cls))
                conf = box.conf
                
                # Draw the rectangle and label.
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                agreement = len(box.model_ids)
                label_text = f"{label} ({conf:.2f}) M={agreement}"
                cv2.putText(image, label_text, (x1, max(y1 - 10, 0)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Build the output filename similar to YOLOv8Inference.
        base_name, _ = os.path.splitext(image_name)
        initial_filename = base_name
        filename = f"{initial_filename}_aggregated_output.jpg"
        os.makedirs(os.path.join(config.IMAGE_MODE, "results"), exist_ok=True)
        save_path = os.path.join(config.IMAGE_MODE, "results", filename)
        cv2.imwrite(save_path, image)
        print(f"Final aggregated output image saved to {save_path}")
        return image

    def classify_and_draw(self, results_dict, image, image_name=config.IMAGE_NAME):
        """
        Combines classification and drawing of aggregated bounding boxes.
        
        Parameters:
            results_dict (dict): Dictionary mapping model_id to YOLOv8 results.
            image (numpy.ndarray): The image on which to draw the boxes.
        
        Returns:
            tuple: (aggregated_results, image_with_boxes)
        """
        aggregated_results = self.classify(results_dict)
        image_with_boxes = self.draw_aggregated_bounding_boxes(image.copy(), aggregated_results, image_name=image_name)
        return aggregated_results
