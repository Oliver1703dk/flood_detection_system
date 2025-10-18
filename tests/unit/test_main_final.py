import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flood_classifier.utils.metadata_utils import merge_metadata


def test_merge_metadata_preserves_video_fields_and_overrides_later_values():
    original = {
        "camera_id": "video_camera_01",
        "timestamp": "2025-10-16T12:14:08.891354",
        "video_timestamp_sec": 10.214,
        "video_file": "flood_video.mp4",
    }
    enriched = {
        "timestamp": "2025-10-16T12:14:08.891354Z",
        "sensor_baseline": {"temperature_baseline": 17.0},
        "sensor_anomalies": {"delta_temperature": -5.0},
    }

    merged = merge_metadata(original, enriched)

    # Existing keys remain while derived metadata is added.
    assert merged["video_timestamp_sec"] == 10.214
    assert merged["video_file"] == "flood_video.mp4"
    # Later metadata overrides earlier duplicates.
    assert merged["timestamp"] == "2025-10-16T12:14:08.891354Z"
    # New keys from enriched metadata are present.
    assert merged["sensor_baseline"] == {"temperature_baseline": 17.0}
    assert merged["sensor_anomalies"] == {"delta_temperature": -5.0}
