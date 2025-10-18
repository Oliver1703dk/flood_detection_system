import os
import re
from typing import Optional

_SEGMENT_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_path_segment(segment: Optional[str], fallback: str = "unknown") -> str:
    """
    Return a filesystem-safe path segment derived from ``segment``.

    The function strips leading/trailing whitespace, replaces path separators,
    and substitutes unsupported characters with underscores.
    """
    if not segment:
        return fallback

    value = str(segment).strip()
    if not value:
        return fallback

    for sep in {os.sep, os.altsep}:
        if sep:
            value = value.replace(sep, "_")

    value = _SEGMENT_PATTERN.sub("_", value)
    if value in {"", ".", ".."}:
        return fallback

    return value
