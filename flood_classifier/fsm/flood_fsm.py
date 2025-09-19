"""Finite State Machine managing flood detection tiers and decisions.

This module coordinates YOLO model tier selection, sensor/image fusion and
optional LLM confirmation to reduce latency while maintaining correctness.
"""
from __future__ import annotations

import base64
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import config

try:  # Lazy import so unit tests can supply fakes without heavy dependencies.
    from yolov8_processor.preprocessing.image_processor import ImageProcessor
except Exception:  # pragma: no cover - optional dependency path
    ImageProcessor = None  # type: ignore

try:  # pragma: no cover - optional dependency path
    from yolov8_processor.inference.yolov8_inference import YOLOv8Inference
    from yolov8_processor.postprocessing.result_formatter import ResultFormatter
except Exception:  # pragma: no cover - optional dependency path
    YOLOv8Inference = None  # type: ignore
    ResultFormatter = None  # type: ignore

try:  # pragma: no cover - optional dependency path
    from yolov8_processor.inference.multi_model_inference import MultiModelInference
except Exception:  # pragma: no cover - allow unit tests to stub
    MultiModelInference = None  # type: ignore

try:  # pragma: no cover - optional dependency path
    from flood_classifier.inference.classifier_both import ClassifierBoth
except Exception:  # pragma: no cover - unit tests can inject substitutes
    ClassifierBoth = None  # type: ignore

try:  # pragma: no cover - optional dependency path
    from flood_classifier.inference.llm_image_classifier import LLMImageClassifier
except Exception:  # pragma: no cover - unit tests can inject substitutes
    LLMImageClassifier = None  # type: ignore


class MotionState(Enum):
    """Motion hint provided by upstream sensors."""

    FAST = "fast"
    SLOW = "slow"
    STOP = "stop"

    @classmethod
    def from_any(cls, value: Optional[str]) -> "MotionState":
        if value is None:
            return MotionState.SLOW
        value = value.strip().lower()
        for member in cls:
            if member.value == value:
                return member
        return MotionState.SLOW


class ModelTier(Enum):
    """Supported YOLO model tiers managed by the FSM."""

    NANO = "nano"
    SMALL = "small"
    MEDIUM = "medium"

    def higher(self) -> "ModelTier":
        order = [ModelTier.NANO, ModelTier.SMALL, ModelTier.MEDIUM]
        idx = order.index(self)
        return order[min(idx + 1, len(order) - 1)]

    def lower(self) -> "ModelTier":
        order = [ModelTier.NANO, ModelTier.SMALL, ModelTier.MEDIUM]
        idx = order.index(self)
        return order[max(idx - 1, 0)]


class FloodState(Enum):
    """FSM states matching the high-level behaviour specification."""

    S0 = "normal"
    S1 = "suspicious"
    S2 = "flood"
    S3 = "ambiguous"
    S5 = "resource"


@dataclass
class FSMParams:
    """Configuration knobs controlling thresholds, counters and resources."""

    threshold_low: float = 0.35
    threshold_high: float = 0.65
    frames_high: int = 3  # M consecutive high frames required for flood
    frames_low: int = 3  # N consecutive low frames required to clear flood
    frames_ambiguous: int = 3  # K consecutive ambiguous frames for escalation
    model_cooldown: int = 5  # C frames between YOLO model switches
    flap_window: int = 8  # W window for oscillation detection
    resource_skip_ratio: int = 3  # Process 1 of N frames while constrained
    accuracy_weight: float = field(
        default_factory=lambda: getattr(config, "IMPORTANCE", {}).get("accuracy", 0.34)
    )
    timeliness_weight: float = field(
        default_factory=lambda: getattr(config, "IMPORTANCE", {}).get("timeliness", 0.33)
    )
    energy_weight: float = field(
        default_factory=lambda: getattr(config, "IMPORTANCE", {}).get("energy", 0.33)
    )
    image_size: Tuple[int, int] = field(default_factory=lambda: getattr(config, "IMAGE_SIZE", (640, 640)))
    llm_enabled: bool = field(default_factory=lambda: getattr(config, "USE_LLM_CONFIRMATION", False))
    llm_model: str = "gpt-4.1-mini"
    tier_paths: Dict[ModelTier, str] = field(
        default_factory=lambda: {
            ModelTier.NANO: "nano/best1.pt",
            ModelTier.SMALL: "small/best1.pt",
            ModelTier.MEDIUM: "medium/best1.pt",
        }
    )


@dataclass
class FrameContext:
    """Inputs required to produce a decision for a single frame."""

    image_b64: Optional[str]
    sensor_data: Dict[str, Any]
    timestamp: Optional[datetime] = None
    motion_hint: Optional[str] = None
    resource_constrained: bool = False
    preprocessed_image: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_motion_state(self) -> MotionState:
        motion = self.metadata.get("motion", self.motion_hint)
        return MotionState.from_any(motion)

    def resolved_timestamp(self) -> datetime:
        if self.timestamp:
            return self.timestamp
        ts = self.metadata.get("timestamp")
        if isinstance(ts, datetime):
            return ts
        return datetime.utcnow()

    def image_bytes(self) -> Optional[bytes]:
        if not self.image_b64:
            return None
        try:
            return base64.b64decode(self.image_b64)
        except Exception:  # pragma: no cover - invalid base64 handled gracefully
            return None


@dataclass
class FrameDecision:
    """Result of evaluating a frame through the FSM."""

    frame_index: int
    state: FloodState
    model_tier: ModelTier
    prediction: int
    scores: Dict[str, float]
    counters: Dict[str, int]
    llm_used: bool
    llm_prediction: Optional[int]
    conflict: bool
    drift: bool
    flapping: bool
    skipped: bool
    model_switched: bool
    tier_requested: ModelTier


class ModelManager:
    """Ensure a single YOLO model stays active and enforce switching rules."""

    def __init__(
        self,
        params: FSMParams,
        load_model: Optional[Callable[[ModelTier, str], Any]] = None,
        image_processor: Optional[ImageProcessor] = None,
        result_formatter: Optional[Any] = None,
        multi_model_inference: Optional[Any] = None,
    ) -> None:
        self.params = params
        self._load_model_cb = load_model or self._default_loader
        self._image_processor = image_processor or (ImageProcessor(params.image_size) if ImageProcessor else None)
        self._formatter = result_formatter or (ResultFormatter() if ResultFormatter else None)
        if multi_model_inference is not None:
            self._multi_inference = multi_model_inference
        elif MultiModelInference is not None:
            try:
                self._multi_inference = MultiModelInference()
            except Exception as exc:
                raise RuntimeError(
                    "Failed to initialize MultiModelInference; provide an instance via multi_model_inference or resolve the underlying issue."
                ) from exc
        else:
            self._multi_inference = None
        if self._multi_inference is None and MultiModelInference is None:
            raise RuntimeError(
                "MultiModelInference unavailable; supply a multi_model_inference instance or install the required YOLO stack."
            )
        self._active_model: Any = None
        self._active_tier: Optional[ModelTier] = None
        self._last_switch_frame: int = -params.model_cooldown

    @property
    def active_tier(self) -> Optional[ModelTier]:
        return self._active_tier

    def _default_loader(self, tier: ModelTier, model_path: str) -> Any:
        if YOLOv8Inference is None:
            raise RuntimeError("YOLOv8Inference unavailable; provide custom load_model callback")
        return YOLOv8Inference(model_filename=model_path, identifier=tier.value)

    def _preprocess(self, ctx: FrameContext) -> Any:
        if ctx.preprocessed_image is not None:
            return ctx.preprocessed_image
        if self._image_processor is None:
            raise RuntimeError("ImageProcessor unavailable; supply preprocessed_image in FrameContext")
        if not ctx.image_b64:
            raise ValueError("FrameContext.image_b64 is required when preprocessed image missing")
        return self._image_processor.preprocess(ctx.image_b64)

    def _format_detections(self, results: Any, tier: ModelTier) -> List[Dict[str, Any]]:
        if self._formatter is None:
            # Tests can provide already formatted detections via custom loader/formatter.
            return results  # type: ignore[return-value]
        formatted = self._formatter.format_results(results)
        aggregated_model_ids: List[Optional[List[str]]] = []
        try:
            for result in results:
                boxes = getattr(result, "boxes", [])
                for box in boxes:
                    mids = getattr(box, "model_ids", None)
                    if mids:
                        aggregated_model_ids.append(sorted(str(mid) for mid in mids))
                    else:
                        aggregated_model_ids.append(None)
        except TypeError:
            aggregated_model_ids = []

        for det, mids in zip(formatted, aggregated_model_ids):
            model_ids = det.setdefault("model_ids", [])
            if mids:
                for mid in mids:
                    if mid not in model_ids:
                        model_ids.append(mid)
        for det in formatted:
            model_ids = det.setdefault("model_ids", [])
            if tier.value not in model_ids:
                model_ids.append(tier.value)
        return formatted

    def _load_tier(self, tier: ModelTier, frame_index: int) -> None:
        if self._multi_inference is not None:
            self._active_model = None
            self._active_tier = tier
            self._last_switch_frame = frame_index
            return
        model_path = self.params.tier_paths.get(tier)
        if not model_path:
            raise ValueError(f"No model path configured for tier {tier}")
        self._active_model = self._load_model_cb(tier, model_path)
        self._active_tier = tier
        self._last_switch_frame = frame_index

    def infer(
        self, ctx: FrameContext, requested_tier: ModelTier, frame_index: int
    ) -> Tuple[List[Dict[str, Any]], ModelTier, bool]:
        switched = False
        if self._active_tier is None:
            self._load_tier(requested_tier, frame_index)
            switched = True
        elif requested_tier != self._active_tier:
            frames_since_switch = frame_index - self._last_switch_frame
            if frames_since_switch >= self.params.model_cooldown:
                self._load_tier(requested_tier, frame_index)
                switched = True
            else:
                requested_tier = self._active_tier
        image = self._preprocess(ctx)
        image_name = ctx.metadata.get("image_name", f"frame-{frame_index}")

        if self._multi_inference is not None:
            results = self._multi_inference.run_all_inference(image, image_name=image_name)
            active_tier = self._active_tier or requested_tier
        else:
            if hasattr(self._active_model, "run_inference"):
                results = self._active_model.run_inference(image, image_name=image_name)
            else:
                # Allow dependency injection in tests where run_inference is a callable.
                results = self._active_model(image)
            active_tier = self._active_tier or requested_tier

        detections = self._format_detections(results, active_tier)
        return detections, active_tier, switched


class FloodFSM:
    """Stateful orchestrator controlling YOLO tier usage and classification."""

    def __init__(
        self,
        params: Optional[FSMParams] = None,
        classifier: Optional[Any] = None,
        model_manager: Optional[ModelManager] = None,
        llm_classifier: Optional[Any] = None,
        now_fn: Callable[[], datetime] = datetime.utcnow,
    ) -> None:
        self.params = params or FSMParams()
        self.classifier = classifier or (ClassifierBoth() if ClassifierBoth else None)
        if self.classifier is None:
            raise RuntimeError("ClassifierBoth unavailable; please provide classifier instance")
        self.model_manager = model_manager or ModelManager(self.params)
        if self.params.llm_enabled:
            self.llm_classifier = llm_classifier or (
                LLMImageClassifier(model=self.params.llm_model) if LLMImageClassifier else None
            )
        else:
            self.llm_classifier = None
        self.now_fn = now_fn

        self.state: FloodState = FloodState.S0
        self._frame_index: int = -1
        self._counters: Dict[str, int] = {"high": 0, "low": 0, "ambiguous": 0, "conflict": 0}
        self._score_history: Deque[float] = deque(maxlen=self.params.flap_window)
        self._last_decision: Optional[FrameDecision] = None
        self._resource_anchor_state: Optional[FloodState] = None
        self._resource_skip_cursor: int = 0
        self._llm_last_used: Dict[FloodState, int] = {}

    def next_state(self, ctx: FrameContext) -> FrameDecision:
        self._frame_index += 1
        motion = ctx.get_motion_state()
        resource_flag = ctx.resource_constrained or bool(ctx.metadata.get("resource_constrained"))
        prev_state = self.state

        if resource_flag and self.state != FloodState.S5:
            self._resource_anchor_state = self.state
            self.state = FloodState.S5
            self._resource_skip_cursor = 0
        elif not resource_flag and self.state == FloodState.S5:
            # allow evaluation below to move to the inferred state
            pass

        if self.state == FloodState.S5 and resource_flag:
            if self._resource_skip_cursor % self.params.resource_skip_ratio != 0:
                self._resource_skip_cursor += 1
                decision = self._reuse_last_decision()
                new_decision = self._finalize_decision(
                    base_decision=decision,
                    frame_index=self._frame_index,
                    state=self.state,
                    model_tier=decision.model_tier,
                    tier_requested=decision.tier_requested,
                    prediction=decision.prediction,
                    scores=decision.scores,
                    counters=self._counters.copy(),
                    llm_used=False,
                    llm_prediction=None,
                    conflict=decision.conflict,
                    drift=decision.drift,
                    flapping=False,
                    skipped=True,
                    model_switched=False,
                )
                self._last_decision = new_decision
                return new_decision
            self._resource_skip_cursor = 1  # current frame processed; start skip cycle next frame

        requested_tier = self._choose_tier(motion)
        detections, model_tier, switched = self.model_manager.infer(ctx, requested_tier, self._frame_index)

        timestamp = ctx.resolved_timestamp()
        scores = self.classifier.classify_flood(
            sensor_data=ctx.sensor_data,
            detection_data=detections,
            image_size=self.params.image_size,
            timestamp=timestamp,
        )
        combined = scores.get("combined_score", 0.0)
        image_score = scores.get("image_score", 0.0)
        sensor_boost = scores.get("sensor_boost", 0.0)
        prediction = scores.get("final_prediction", 0)

        conflict = self._detect_conflict(combined, image_score, sensor_boost)
        drift = self._detect_drift(ctx.sensor_data)

        ambiguous = self.params.threshold_low <= combined < self.params.threshold_high or conflict
        self._update_counters(combined, ambiguous, conflict)
        self._score_history.append(combined)
        flapping = self._is_flapping()

        llm_used = False
        llm_prediction: Optional[int] = None
        if (
            self.llm_classifier
            and prev_state in (FloodState.S1, FloodState.S3)
            and motion == MotionState.STOP
            and self._counters["ambiguous"] >= self.params.frames_ambiguous
            and self._llm_last_used.get(prev_state, -1) < self._frame_index - self.params.frames_ambiguous
            and self.state != FloodState.S5
        ):
            image_bytes = ctx.image_bytes()
            if image_bytes is not None:
                try:
                    llm_prediction = int(self.llm_classifier.classify_flood(image_bytes))
                    llm_used = True
                    self._llm_last_used[prev_state] = self._frame_index
                    prediction = llm_prediction
                except Exception:
                    # LLM failures are non-fatal; remain with existing prediction.
                    llm_used = False
                    llm_prediction = None

        next_state = self._determine_next_state(
            current_state=self.state,
            combined=combined,
            resource_flag=resource_flag,
            ambiguous=ambiguous,
            flapping=flapping,
        )
        self.state = next_state
        if self.state != FloodState.S5:
            self._resource_anchor_state = None
            self._resource_skip_cursor = 0

        decision = self._finalize_decision(
            base_decision=None,
            frame_index=self._frame_index,
            state=self.state,
            model_tier=model_tier,
            tier_requested=requested_tier,
            prediction=prediction,
            scores={
                "combined_score": combined,
                "image_score": image_score,
                "sensor_boost": sensor_boost,
            },
            counters=self._counters.copy(),
            llm_used=llm_used,
            llm_prediction=llm_prediction,
            conflict=conflict,
            drift=drift,
            flapping=flapping,
            skipped=False,
            model_switched=switched,
        )
        self._last_decision = decision
        return decision

    def _reuse_last_decision(self) -> FrameDecision:
        if self._last_decision is not None:
            return self._last_decision
        # Construct a neutral decision for the very first skipped frame.
        return FrameDecision(
            frame_index=self._frame_index,
            state=self.state,
            model_tier=self.model_manager.active_tier or ModelTier.NANO,
            prediction=0,
            scores={"combined_score": 0.0, "image_score": 0.0, "sensor_boost": 0.0},
            counters=self._counters.copy(),
            llm_used=False,
            llm_prediction=None,
            conflict=False,
            drift=False,
            flapping=False,
            skipped=True,
            model_switched=False,
            tier_requested=self.model_manager.active_tier or ModelTier.NANO,
        )

    def _finalize_decision(
        self,
        base_decision: Optional[FrameDecision],
        frame_index: int,
        state: FloodState,
        model_tier: ModelTier,
        tier_requested: ModelTier,
        prediction: int,
        scores: Dict[str, float],
        counters: Dict[str, int],
        llm_used: bool,
        llm_prediction: Optional[int],
        conflict: bool,
        drift: bool,
        flapping: bool,
        skipped: bool,
        model_switched: bool,
    ) -> FrameDecision:
        if base_decision and skipped:
            return FrameDecision(
                frame_index=frame_index,
                state=state,
                model_tier=model_tier,
                prediction=base_decision.prediction,
                scores=base_decision.scores,
                counters=counters,
                llm_used=llm_used,
                llm_prediction=llm_prediction,
                conflict=conflict,
                drift=drift,
                flapping=flapping,
                skipped=True,
                model_switched=model_switched,
                tier_requested=tier_requested,
            )
        return FrameDecision(
            frame_index=frame_index,
            state=state,
            model_tier=model_tier,
            prediction=prediction,
            scores=scores,
            counters=counters,
            llm_used=llm_used,
            llm_prediction=llm_prediction,
            conflict=conflict,
            drift=drift,
            flapping=flapping,
            skipped=skipped,
            model_switched=model_switched,
            tier_requested=tier_requested,
        )

    def _choose_tier(self, motion: MotionState) -> ModelTier:
        anchor_state = self._resource_anchor_state if self.state == FloodState.S5 else self.state
        if anchor_state == FloodState.S0:
            base_tier = ModelTier.NANO
        elif anchor_state == FloodState.S1:
            if motion == MotionState.FAST:
                base_tier = ModelTier.SMALL
            else:
                acc = self.params.accuracy_weight
                if acc >= max(self.params.timeliness_weight, self.params.energy_weight):
                    base_tier = ModelTier.MEDIUM
                else:
                    base_tier = ModelTier.SMALL
        elif anchor_state == FloodState.S2:
            base_tier = ModelTier.SMALL if motion == MotionState.FAST else ModelTier.MEDIUM
        elif anchor_state == FloodState.S3:
            current = self.model_manager.active_tier or ModelTier.SMALL
            base_tier = current.higher()
        else:  # S5 fallback and any other unexpected state
            base_tier = self.model_manager.active_tier or ModelTier.NANO

        if self.state == FloodState.S5:
            degraded = base_tier.lower()
            if anchor_state == FloodState.S2 and degraded == ModelTier.NANO:
                degraded = ModelTier.SMALL
            return degraded
        return base_tier

    def _detect_conflict(self, combined: float, image_score: float, sensor_boost: float) -> bool:
        # Conflict when sensor boost strongly suggests flood but image score is low, or vice versa.
        high = self.params.threshold_high
        low = self.params.threshold_low
        image_high = image_score >= high
        image_low = image_score < low
        combined_high = combined >= high
        combined_low = combined < low
        sensor_high = sensor_boost > 0.2
        sensor_low = sensor_boost < 0.02
        return (
            (sensor_high and image_low)
            or (image_high and sensor_low)
            or (combined_high and image_low)
            or (combined_low and sensor_high)
        )

    @staticmethod
    def _detect_drift(sensor_data: Dict[str, Any]) -> bool:
        return bool(
            sensor_data.get("drift")
            or sensor_data.get("sensor_drift")
            or sensor_data.get("drift_flag")
        )

    def _update_counters(self, combined: float, ambiguous: bool, conflict: bool) -> None:
        if combined >= self.params.threshold_high:
            self._counters["high"] += 1
        else:
            self._counters["high"] = 0

        if combined < self.params.threshold_low:
            self._counters["low"] += 1
        else:
            self._counters["low"] = 0

        if ambiguous:
            self._counters["ambiguous"] += 1
        else:
            self._counters["ambiguous"] = 0

        if conflict:
            self._counters["conflict"] += 1
        else:
            self._counters["conflict"] = 0

    def _is_flapping(self) -> bool:
        if len(self._score_history) < 3:
            return False
        buckets: List[int] = []
        for val in self._score_history:
            if val >= self.params.threshold_high:
                buckets.append(1)
            elif val < self.params.threshold_low:
                buckets.append(-1)
            else:
                buckets.append(0)
        transitions = 0
        prev = buckets[0]
        for current in buckets[1:]:
            if current == 0 or prev == 0:
                prev = current
                continue
            if current != prev:
                transitions += 1
            prev = current
        return transitions >= 2 and len(self._score_history) >= min(self.params.flap_window, 4)

    def _determine_next_state(
        self,
        current_state: FloodState,
        combined: float,
        resource_flag: bool,
        ambiguous: bool,
        flapping: bool,
    ) -> FloodState:
        if current_state == FloodState.S5:
            if resource_flag:
                return FloodState.S5
            # Recovery path when resources return.
            if combined >= self.params.threshold_high:
                return FloodState.S2
            if combined < self.params.threshold_low:
                return FloodState.S0
            if self._counters["conflict"] >= self.params.frames_ambiguous:
                return FloodState.S3
            return FloodState.S1

        if current_state == FloodState.S0:
            if ambiguous or self._counters["conflict"] >= self.params.frames_ambiguous:
                return FloodState.S1
            return FloodState.S0

        if current_state == FloodState.S1:
            if self._counters["high"] >= self.params.frames_high:
                return FloodState.S2
            if self._counters["low"] >= self.params.frames_low:
                return FloodState.S0
            if self._counters["conflict"] >= self.params.frames_ambiguous:
                return FloodState.S3
            return FloodState.S1

        if current_state == FloodState.S2:
            if self._counters["low"] >= self.params.frames_low:
                return FloodState.S0
            if flapping:
                return FloodState.S3
            return FloodState.S2

        if current_state == FloodState.S3:
            if self._counters["high"] >= self.params.frames_high:
                return FloodState.S2
            if self._counters["low"] >= self.params.frames_low:
                return FloodState.S0
            if resource_flag:
                return FloodState.S5
            return FloodState.S3

        return FloodState.S0


def decision_to_dict(decision: FrameDecision) -> Dict[str, Any]:
    """Convert a :class:`FrameDecision` into a JSON-serialisable dict."""
    return {
        "frame_index": decision.frame_index,
        "state": decision.state.value,
        "model_tier": decision.model_tier.value,
        "prediction": decision.prediction,
        "scores": decision.scores,
        "counters": decision.counters,
        "llm_used": decision.llm_used,
        "llm_prediction": decision.llm_prediction,
        "conflict": decision.conflict,
        "drift": decision.drift,
        "flapping": decision.flapping,
        "skipped": decision.skipped,
        "model_switched": decision.model_switched,
        "tier_requested": decision.tier_requested.value,
    }
