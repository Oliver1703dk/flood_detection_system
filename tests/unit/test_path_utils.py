import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from path_utils import sanitize_path_segment


def test_sanitize_path_segment_replaces_separators_and_invalid_chars():
    assert sanitize_path_segment("folder/name.png") == "folder_name.png"
    assert sanitize_path_segment("strange*value?") == "strange_value_"


def test_sanitize_path_segment_handles_blank_values():
    assert sanitize_path_segment("") == "unknown"
    assert sanitize_path_segment(None, fallback="fallback") == "fallback"


def test_sanitize_path_segment_preserves_extension_and_valid_chars():
    segment = "flood_video_20251005_150618.mp4"
    assert sanitize_path_segment(segment) == segment
