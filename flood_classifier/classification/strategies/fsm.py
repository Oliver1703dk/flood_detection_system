"""Classification strategy that routes frames through the FloodFSM."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from ..strategy import ClassificationStrategy
from flood_classifier.fsm.flood_fsm import (
    FloodFSM,
    FrameContext,
    FrameDecision,
    decision_to_dict,
)
from flood_classifier.inference.classifier_both import ClassifierBoth

_default_fsm: Optional[FloodFSM] = None


def _get_default_fsm(vision_only: bool = False) -> FloodFSM:
    global _default_fsm
    if _default_fsm is None:
        # Create classifier with sensor fusion disabled if vision_only mode
        if vision_only:
            classifier = ClassifierBoth(disable_sensor_fusion=True)
            _default_fsm = FloodFSM(classifier=classifier)
        else:
            _default_fsm = FloodFSM()
    return _default_fsm


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


class FSMStrategy(ClassificationStrategy):
    """Classification strategy backed by :class:`FloodFSM`."""

    def __init__(self, fsm: Optional[FloodFSM] = None, vision_only: bool = False) -> None:
        # Read DISABLE_SENSOR_FUSION from config if vision_only not explicitly set
        import config
        if not vision_only and hasattr(config, 'DISABLE_SENSOR_FUSION'):
            vision_only = bool(getattr(config, 'DISABLE_SENSOR_FUSION', False))
        self._fsm = fsm if fsm is not None else _get_default_fsm(vision_only=vision_only)

    def classify(self, detection_results, message):  # type: ignore[override]
        """Ignore external detections and delegate to the FSM."""
        ctx = self._build_context(message)
        decision = self._fsm.next_state(ctx)
        return decision

    def _build_context(self, message: Dict[str, Any]) -> FrameContext:
        metadata = message.get("metadata", {})
        ts = message.get("timestamp") or metadata.get("timestamp")
        timestamp = _parse_timestamp(ts)
        motion = metadata.get("motion") or message.get("motion")
        resource_flag = bool(metadata.get("resource_constrained") or message.get("resource_constrained"))
        image_data = message.get("image_data")
        return FrameContext(
            image_b64=image_data,
            sensor_data=message.get("sensor_data", {}),
            timestamp=timestamp,
            motion_hint=motion,
            resource_constrained=resource_flag,
            metadata=metadata,
        )


def classify_frame(message: Dict[str, Any]) -> FrameDecision:
    """Convenience wrapper for stateless callers."""
    return FSMStrategy().classify([], message)


__all__ = ["FSMStrategy", "classify_frame", "decision_to_dict"]
