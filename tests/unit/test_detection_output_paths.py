import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("cv2")

import config
from yolov8_processor.classifier.yolov8_final_classifier import (
    AggregatedBox,
    AggregatedResult,
    YOLOv8FinalClassifier,
)


def test_aggregated_detections_route_to_video_folder(tmp_path, monkeypatch):
    # Ensure detection output is redirected to a temporary folder for the test.
    monkeypatch.setattr(config, "DETECTION_OUTPUT_DIR", tmp_path)

    classifier = YOLOv8FinalClassifier()
    box = AggregatedBox([50, 50, 20, 20], conf=0.8, cls=0, model_ids={"1"})
    aggregated = AggregatedResult(names={0: "water"}, boxes=[box])

    saved_paths: list[Path] = []

    def fake_imwrite(path, image):
        saved_paths.append(Path(path))
        return True

    monkeypatch.setattr("yolov8_processor.classifier.yolov8_final_classifier.cv2.imwrite", fake_imwrite)

    image = np.zeros((100, 100, 3), dtype=np.uint8)

    classifier.draw_aggregated_bounding_boxes(
        image,
        [aggregated],
        run_id="123",
        metadata={"video_file": "clips/flood_clip.mp4"},
    )

    assert saved_paths, "Expected a detection image to be written"
    expected_dir = tmp_path / "clips_flood_clip.mp4"
    assert saved_paths[0].parent == expected_dir
