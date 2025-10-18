import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flood_classifier.postprocessing.data_result_saver import DataResultsSaver


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def test_save_default_path_without_video_file(tmp_path):
    storage_dir = tmp_path / "data_results"
    video_dir = tmp_path / "video_results"
    saver = DataResultsSaver(storage_dir=str(storage_dir), video_storage_dir=str(video_dir))

    payload = {
        "sensor_data": {"temperature": 21},
        "metadata": {"camera_id": "camera-01"},
    }

    saver.save(payload, {"state": "S0"})

    subdirs = list(storage_dir.iterdir())
    assert len(subdirs) == 1, "Expected a single date folder in data_results"
    saved_file = next(subdirs[0].iterdir())
    saved_data = _load_json(saved_file)

    assert saved_data["metadata"]["camera_id"] == "camera-01"
    assert not any(video_dir.iterdir()), "Video results directory should remain empty"


def test_save_video_path_when_video_file_present(tmp_path):
    storage_dir = tmp_path / "data_results"
    video_dir = tmp_path / "video_results"
    saver = DataResultsSaver(storage_dir=str(storage_dir), video_storage_dir=str(video_dir))

    payload = {
        "sensor_data": {"temperature": 19},
        "metadata": {
            "camera_id": "cam/02",
            "video_file": "clips/flood_clip.mp4",
        },
    }

    saver.save(payload, {"state": "S1"})

    # data_results should remain empty (no per-date folder created)
    assert not any(storage_dir.iterdir()), "No files should be stored under data_results for video payloads"

    video_subdirs = list(video_dir.iterdir())
    assert len(video_subdirs) == 1
    assert video_subdirs[0].name == "clips_flood_clip.mp4"

    saved_file = next(video_subdirs[0].iterdir())
    saved_data = _load_json(saved_file)
    assert saved_data["metadata"]["video_file"] == "clips/flood_clip.mp4"
    assert saved_data["metadata"]["camera_id"] == "cam/02"
