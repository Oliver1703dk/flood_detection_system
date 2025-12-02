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
from time import perf_counter
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import config

try:  # Import energy tracker
    from flood_classifier.utils.energy_tracker import get_energy_tracker
except Exception:  # pragma: no cover - optional dependency path
    get_energy_tracker = None  # type: ignore

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

try:  # pragma: no cover - optional dependency path
    from flood_classifier.inference.dispatcher import InferenceDispatcher, MQTTJetsonBackend
except Exception:  # pragma: no cover - dispatcher relies on optional deps
    InferenceDispatcher = None  # type: ignore
    MQTTJetsonBackend = None  # type: ignore

try:  # pragma: no cover - optional dependency path
    from flood_classifier.utils.energy_tracker import get_energy_tracker
except Exception:  # pragma: no cover - unit tests can stub
    get_energy_tracker = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from flood_classifier.inference.dispatcher import InferenceDispatcher


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
    LARGE = "large"

    def higher(self) -> "ModelTier":
        order = [ModelTier.NANO, ModelTier.SMALL, ModelTier.MEDIUM, ModelTier.LARGE]
        idx = order.index(self)
        return order[min(idx + 1, len(order) - 1)]

    def lower(self) -> "ModelTier":
        order = [ModelTier.NANO, ModelTier.SMALL, ModelTier.MEDIUM, ModelTier.LARGE]
        idx = order.index(self)
        return order[max(idx - 1, 0)]


class FloodState(Enum):
    """FSM states matching the high-level behaviour specification."""

    S0 = "S0"
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    S5 = "S5"


@dataclass
class FSMParams:
    """Configuration knobs controlling thresholds, counters and resources."""

    threshold_low: float = field(
        default_factory=lambda: getattr(config, "FSM_DEFAULTS", {}).get("threshold_low", 0.35)
    )
    threshold_high: float = field(
        default_factory=lambda: getattr(config, "FSM_DEFAULTS", {}).get("threshold_high", 0.65)
    )
    frames_high: int = field(
        default_factory=lambda: getattr(config, "FSM_DEFAULTS", {}).get("frames_high", 3)
    )  # M consecutive high frames required for flood
    frames_low: int = field(
        default_factory=lambda: getattr(config, "FSM_DEFAULTS", {}).get("frames_low", 3)
    )  # N consecutive low frames required to clear flood
    frames_ambiguous: int = field(
        default_factory=lambda: getattr(config, "FSM_DEFAULTS", {}).get("frames_ambiguous", 3)
    )  # K consecutive ambiguous frames for escalation
    model_cooldown: int = field(
        default_factory=lambda: getattr(config, "FSM_DEFAULTS", {}).get("model_cooldown", 1)
    )  # C frames between YOLO model switches
    flap_window: int = field(
        default_factory=lambda: getattr(config, "FSM_DEFAULTS", {}).get("flap_window", 8)
    )  # W window for oscillation detection
    resource_skip_ratio: int = field(
        default_factory=lambda: getattr(config, "FSM_DEFAULTS", {}).get("resource_skip_ratio", 3)
    )  # Process 1 of N frames while constrained
    image_size: Tuple[int, int] = field(default_factory=lambda: getattr(config, "IMAGE_SIZE", (640, 640)))
    llm_enabled: bool = field(default_factory=lambda: getattr(config, "USE_LLM_CONFIRMATION", False))
    llm_model: str = "gpt-4.1-mini"
    tier_paths: Dict[ModelTier, str] = field(
        default_factory=lambda: (
            {
                ModelTier.NANO: "baseline/nano/best1.pt",
                ModelTier.SMALL: "baseline/small/best1.pt",
                ModelTier.MEDIUM: "baseline/medium/best1.pt",
                ModelTier.LARGE: "baseline/large/best1.pt",
            }
            if getattr(config, "USE_BASELINE_MODELS", False)
            else {
                ModelTier.NANO: "nano/best1.pt",
                ModelTier.SMALL: "small/best1.pt",
                ModelTier.MEDIUM: "medium/best1.pt",
                ModelTier.LARGE: "large/best1.pt",
            }
        )
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
    backend: str = "local"
    backend_info: Dict[str, Any] = field(default_factory=dict)
    llm_backend_info: Dict[str, Any] = field(default_factory=dict)
    s2_llm_confirmed: bool = False
    timing: Dict[str, float] = field(default_factory=dict)


class ModelManager:
    """Ensure a single YOLO model stays active and enforce switching rules."""

    def __init__(
        self,
        params: FSMParams,
        load_model: Optional[Callable[[ModelTier, str], Any]] = None,
        image_processor: Optional[ImageProcessor] = None,
        result_formatter: Optional[Any] = None,
        multi_model_inference: Optional[Any] = None,
        dispatcher: Optional["InferenceDispatcher"] = None,
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
        self._dispatcher = dispatcher
        # Model cache to keep models in memory for faster switching
        self._model_cache: Dict[ModelTier, Any] = {}
        # Track first inference per tier for cold-start detection
        self._tier_first_inference: Dict[ModelTier, bool] = {}
        # Track model switch overhead timing
        self._switch_overhead_s: Optional[float] = None

    @property
    def active_tier(self) -> Optional[ModelTier]:
        return self._active_tier

    def is_tier_available_locally(self, tier: ModelTier) -> bool:
        """Check if a model tier can be loaded locally.
        
        Returns True if the tier has a configured path in tier_paths.
        This allows the system to determine if a fallback to local inference
        is possible when remote inference fails.
        
        Args:
            tier: The ModelTier to check availability for
            
        Returns:
            True if the tier is configured for local execution, False otherwise
        """
        return tier in self.params.tier_paths and self.params.tier_paths[tier] is not None

    def prewarm_models(self, tiers: Optional[List[ModelTier]] = None) -> Dict[ModelTier, float]:
        """Pre-load models at startup to avoid cold-start delays.
        
        Also runs dummy inference to fully initialize models and eliminate cold-start delays.
        Only prewarms nano and small models by default (Pi-local models).
        Medium and large models run on Jetson and don't need Pi prewarming.
        
        Args:
            tiers: List of tiers to prewarm. If None, prewarm only nano and small tiers (Pi-local).
            
        Returns:
            Dictionary mapping tier to load time in seconds
        """
        if tiers is None:
            # Only prewarm nano and small models (Pi-local models)
            # Medium and large models run on Jetson and don't need Pi prewarming
            tiers = [ModelTier.NANO, ModelTier.SMALL]
        
        load_times: Dict[ModelTier, float] = {}
        
        for tier in tiers:
            if not self.is_tier_available_locally(tier):
                continue
            
            if tier in self._model_cache:
                print(f"Model {tier.value} already loaded, skipping prewarm")
                continue
            
            try:
                load_start = perf_counter()
                model_path = self.params.tier_paths[tier]
                model = self._load_model_cb(tier, model_path)
                self._model_cache[tier] = model
                load_time = perf_counter() - load_start
                load_times[tier] = load_time
                print(f"✅ Pre-warmed {tier.value} model in {load_time:.3f}s")
                
                # Run dummy inference to fully initialize the model
                try:
                    warmup_start = perf_counter()
                    image_size = self.params.image_size
                    dummy_image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
                    print(f"🔥 Running warmup inference for {tier.value} model...")
                    # Direct model call to trigger full initialization
                    if hasattr(model, 'model'):
                        _ = model.model(dummy_image)
                    elif hasattr(model, 'run_inference'):
                        # For MultiModelInference compatibility
                        _ = model.run_inference(dummy_image)
                    warmup_time = perf_counter() - warmup_start
                    print(f"✅ Warmup inference complete for {tier.value} in {warmup_time:.3f}s")
                except Exception as warmup_exc:
                    print(f"⚠️ Warmup inference failed for {tier.value}: {warmup_exc}")
                    print(f"   (Model loaded but may have cold-start on first real inference)")
                    
            except Exception as exc:
                print(f"⚠️ Failed to pre-warm {tier.value} model: {exc}")
        
        return load_times

    def _is_cold_start(self, tier: ModelTier) -> bool:
        """Check if this is the first inference for a tier (cold start)."""
        return tier not in self._tier_first_inference or not self._tier_first_inference[tier]

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
        """Load a model tier, using cache if available to reduce switch overhead."""
        if self._multi_inference is not None:
            self._active_model = None
            self._active_tier = tier
            self._last_switch_frame = frame_index
            return
        
        model_path = self.params.tier_paths.get(tier)
        if not model_path:
            raise ValueError(f"No model path configured for tier {tier}")
        
        # Track model switch overhead
        switch_start = perf_counter()
        
        # Check cache first to avoid reloading
        if tier in self._model_cache:
            self._active_model = self._model_cache[tier]
            print(f"🔄 Switched to cached {tier.value} model (from cache)")
        else:
            # Load model and cache it
            self._active_model = self._load_model_cb(tier, model_path)
            self._model_cache[tier] = self._active_model
            print(f"🔄 Loaded and cached {tier.value} model")
        
        switch_overhead = perf_counter() - switch_start
        self._switch_overhead_s = switch_overhead
        
        self._active_tier = tier
        self._last_switch_frame = frame_index

    def infer(
        self,
        ctx: FrameContext,
        requested_tier: ModelTier,
        frame_index: int,
        fsm_state: "FloodState",
        motion: Optional[MotionState] = None,
    ) -> Tuple[List[Dict[str, Any]], ModelTier, bool, Dict[str, Any]]:
        backend_meta: Dict[str, Any] = {"backend": "local", "metadata": {}}

        if self._dispatcher is not None:
            print(f"🔍 [ModelManager] infer: Attempting remote inference - state={fsm_state.value}, tier={requested_tier.value}, frame={frame_index}")
            try:
                remote_result = self._dispatcher.try_remote_yolo(
                    state=fsm_state,
                    requested_tier=requested_tier,
                    frame_index=frame_index,
                    ctx=ctx,
                    motion=motion,
                )
                if remote_result is not None:
                    print(f"✅ [ModelManager] infer: Remote inference successful - backend={remote_result.backend}")
                else:
                    print(f"🔍 [ModelManager] infer: Remote inference returned None (not routed or failed)")
            except Exception as exc:
                # Check if tier is available locally before falling back
                if not self.is_tier_available_locally(requested_tier):
                    error_msg = (
                        f"Cannot run {requested_tier.value} model: remote backend unavailable "
                        f"and tier not configured for local execution. Error: {exc}"
                    )
                    print(f"⚠️ ERROR: {error_msg}")
                    raise RuntimeError(error_msg) from exc
                
                print(f"⚠️ WARNING: Remote inference failed ({exc}); falling back to local {requested_tier.value} model.")
                remote_result = None

            if remote_result is not None:
                backend_meta = {
                    "backend": remote_result.backend,
                    "metadata": remote_result.metadata,
                }
                return remote_result.detections, remote_result.tier, remote_result.switched, backend_meta

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

        inference_latency: Optional[float] = None
        inference_start_wall = datetime.utcnow()
        yolo_energy = None
        if self._multi_inference is not None:
            inference_start = perf_counter()
            results = self._multi_inference.run_all_inference(
                image,
                image_name=image_name,
                metadata=ctx.metadata,
            )
            inference_latency = perf_counter() - inference_start
            active_tier = self._active_tier or requested_tier
            # Extract YOLO energy metrics for local runs only
            if backend_meta["backend"] == "local":
                yolo_energy = self._multi_inference.last_energy_metrics
        else:
            inference_start = perf_counter()
            if hasattr(self._active_model, "run_inference"):
                results = self._active_model.run_inference(
                    image,
                    image_name=image_name,
                    metadata=ctx.metadata,
                )
            else:
                # Allow dependency injection in tests where run_inference is a callable.
                results = self._active_model(image)
            inference_latency = perf_counter() - inference_start
            active_tier = self._active_tier or requested_tier

        detections = self._format_detections(results, active_tier)
        if inference_latency is not None:
            metadata = backend_meta.setdefault("metadata", {})
            metadata["latency_s"] = inference_latency
            metadata["started_at"] = inference_start_wall.isoformat()
            metadata["completed_at"] = datetime.utcnow().isoformat()
            timing_block = metadata.setdefault("timing", {})
            timing_block.setdefault("inference_s", inference_latency)
            
            # Track cold-start vs warm inference
            is_cold = self._is_cold_start(active_tier)
            timing_block["inference_cold_start"] = 1.0 if is_cold else 0.0
            if is_cold:
                self._tier_first_inference[active_tier] = True
                print(f"❄️ Cold start detected for {active_tier.value} model (inference: {inference_latency:.3f}s)")
            
            # Track model switch overhead if a switch occurred
            if switched and self._switch_overhead_s is not None:
                timing_block["model_switch_overhead_s"] = self._switch_overhead_s
                print(f"⏱️ Model switch overhead: {self._switch_overhead_s:.3f}s")
            
            # Add YOLO energy to metadata only for local runs
            if yolo_energy and backend_meta["backend"] == "local":
                if isinstance(yolo_energy.get("energy_j"), (int, float)):
                    metadata["energy"] = {"energy_j": float(yolo_energy["energy_j"])}
                    timing_block["yolo_energy_j"] = float(yolo_energy["energy_j"])
                if isinstance(yolo_energy.get("power_w"), (int, float)):
                    timing_block["yolo_power_w"] = float(yolo_energy["power_w"])
                if isinstance(yolo_energy.get("cpu_util_%"), (int, float)):
                    timing_block["yolo_cpu_util_%"] = float(yolo_energy["cpu_util_%"])
        return detections, active_tier, switched, backend_meta


class FloodFSM:
    """Stateful orchestrator controlling YOLO tier usage and classification."""

    def __init__(
        self,
        params: Optional[FSMParams] = None,
        classifier: Optional[Any] = None,
        model_manager: Optional[ModelManager] = None,
        dispatcher: Optional["InferenceDispatcher"] = None,
        now_fn: Callable[[], datetime] = datetime.utcnow,
    ) -> None:
        self.params = params or FSMParams()
        self.classifier = classifier or (ClassifierBoth() if ClassifierBoth else None)
        if self.classifier is None:
            raise RuntimeError("ClassifierBoth unavailable; please provide classifier instance")
        self.dispatcher = dispatcher or self._build_dispatcher()
        self.model_manager = model_manager or ModelManager(self.params, dispatcher=self.dispatcher)
        self.llm_enabled = bool(self.params.llm_enabled and self.dispatcher)
        self.now_fn = now_fn

        self.state: FloodState = FloodState.S0
        self._frame_index: int = -1
        self._counters: Dict[str, int] = {
            "high": 0,
            "low": 0,
            "ambiguous": 0,
            "conflict": 0,
            "mid": 0,
        }
        self._score_history: Deque[float] = deque(maxlen=self.params.flap_window)
        self._last_decision: Optional[FrameDecision] = None
        self._resource_anchor_state: Optional[FloodState] = None
        self._resource_skip_cursor: int = 0
        self._llm_last_used: Dict[FloodState, int] = {}
        self._s2_llm_confirmed: bool = False
        self._s2_last_confirm_frame: int = -1
        # Track previous frame's classification state (high or ambiguous) for weighted counting
        self._prev_was_suspicious: bool = False  # True if previous was high or ambiguous
        
        # Pre-warm models at startup if enabled (after model_manager is created)
        if getattr(config, "PREWARM_MODELS", True):
            self._prewarm_models()

    def _prewarm_models(self) -> None:
        """Pre-warm models at startup to avoid cold-start delays."""
        print("\n🔥 Pre-warming models...")
        load_times = self.model_manager.prewarm_models()
        if load_times:
            total_time = sum(load_times.values())
            print(f"✅ Pre-warmed {len(load_times)} model(s) in {total_time:.3f}s total")
        else:
            print("ℹ️ No models to pre-warm (all models may be remote-only)")

    def _build_dispatcher(self) -> Optional["InferenceDispatcher"]:
        # Get available local tiers from params (available before model_manager is created)
        available_tiers = {tier.value for tier in self.params.tier_paths.keys()}
        print(f"🔍 [FSM] _build_dispatcher: available_local_tiers={available_tiers}")
        
        if InferenceDispatcher is None or MQTTJetsonBackend is None:
            print(f"🔍 [FSM] _build_dispatcher: InferenceDispatcher or MQTTJetsonBackend is None, returning None")
            return None
        try:
            print(f"🔍 [FSM] _build_dispatcher: Attempting to create MQTTJetsonBackend...")
            remote_backend = MQTTJetsonBackend()
            print(f"🔍 [FSM] _build_dispatcher: MQTTJetsonBackend created successfully")
            # Check initial health
            initial_health = remote_backend.is_healthy()
            print(f"🔍 [FSM] _build_dispatcher: Initial backend health check: {initial_health}")
        except Exception as exc:
            print(f"⚠️ [FSM] Remote inference disabled ({exc}). Running local-only mode.")
            import traceback
            traceback.print_exc()
            return None
        dispatcher = InferenceDispatcher(
            remote_backend=remote_backend,
            available_local_tiers=available_tiers
        )
        print(f"🔍 [FSM] _build_dispatcher: InferenceDispatcher created with routing={dispatcher.routing}")
        return dispatcher

    def next_state(self, ctx: FrameContext) -> FrameDecision:
        fsm_timing: Dict[str, float] = {}
        fsm_start = perf_counter()
        
        self._frame_index += 1
        
        # Time state transition logic
        state_transition_start = perf_counter()
        motion = ctx.get_motion_state()
        resource_flag = ctx.resource_constrained or bool(ctx.metadata.get("resource_constrained"))
        prev_state = self.state

        timestamp = ctx.resolved_timestamp()
        baseline_calc = getattr(self.classifier, "baseline_calculator", None)
        baseline_lookup_start = perf_counter()
        baseline = baseline_calc.get_baseline_for_time(timestamp) if baseline_calc else None
        if baseline_calc is not None:
            fsm_timing["fsm_baseline_lookup_s"] = perf_counter() - baseline_lookup_start
        if baseline is not None:
            ctx.metadata["sensor_baseline"] = baseline
        ctx.metadata.setdefault("sensor_data", ctx.sensor_data)
        ctx.metadata.setdefault("timestamp", timestamp.isoformat())
        state_transition_duration = perf_counter() - state_transition_start
        fsm_timing["fsm_state_transition_s"] = state_transition_duration

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
                    s2_llm_confirmed=self._s2_llm_confirmed,
                )
                self._last_decision = new_decision
                return new_decision
            self._resource_skip_cursor = 1  # current frame processed; start skip cycle next frame

        tier_select_start = perf_counter()
        requested_tier = self._choose_tier(motion, resource_flag=resource_flag)
        fsm_timing["fsm_tier_selection_s"] = perf_counter() - tier_select_start

        infer_dispatch_start = perf_counter()
        detections, model_tier, switched, backend_meta = self.model_manager.infer(
            ctx,
            requested_tier,
            self._frame_index,
            prev_state,
            motion=motion,
        )
        fsm_timing["fsm_infer_dispatch_s"] = perf_counter() - infer_dispatch_start
        if switched:
            fsm_timing["fsm_model_switched_flag"] = 1.0
        backend_name = backend_meta.get("backend", "local")
        backend_info = backend_meta.get("metadata", {})
        timing_info: Dict[str, float] = {}
        if isinstance(backend_info, dict):
            latency_val = backend_info.get("latency_s")
            if isinstance(latency_val, (int, float)):
                timing_info["inference_s"] = float(latency_val)
            backend_timing = backend_info.get("timing") if isinstance(backend_info, dict) else None
            if isinstance(backend_timing, dict):
                for key, value in backend_timing.items():
                    if isinstance(value, (int, float)):
                        timing_info[f"backend_{key}"] = float(value)

        # Track energy for FSM logic (classification, counters, decision logic)
        fsm_energy_metrics = {}
        classification_duration = 0.0
        classification_measure_duration = 0.0
        sensor_prediction = "neutral"  # Default value
        energy_tracking_enabled = bool(getattr(config, "ENABLE_FSM_ENERGY_TRACKING", True))
        if get_energy_tracker is not None and energy_tracking_enabled:
            energy_tracker = get_energy_tracker()
            energy_measure_start = perf_counter()
            with energy_tracker.measure("fsm_classification") as metrics:
                classification_phase_start = perf_counter()
                scores = self.classifier.classify_flood(
                    sensor_data=ctx.sensor_data,
                    detection_data=detections,
                    image_size=self.params.image_size,
                    timestamp=timestamp,
                )
                combined = scores.get("combined_score", 0.0)
                image_score = scores.get("image_score", 0.0)
                sensor_boost = scores.get("sensor_boost", 0.0)
                sensor_prediction = scores.get("sensor_prediction", "neutral")
                prediction = scores.get("final_prediction", 0)

                conflict = self._detect_conflict(combined, image_score, sensor_boost)
                drift = self._detect_drift(ctx.sensor_data)

                mid_band = self.params.threshold_low <= combined < self.params.threshold_high
                ambiguous = mid_band or conflict
                self._update_counters(combined, ambiguous, conflict, mid_band)
                self._score_history.append(combined)
                flapping = self._is_flapping()

                ctx.metadata["sensor_anomalies"] = scores.get("anomalies")
                classification_duration = perf_counter() - classification_phase_start
            fsm_energy_metrics.update(metrics)
            classification_measure_duration = perf_counter() - energy_measure_start
        else:
            # Fallback when energy tracker not available
            classification_phase_start = perf_counter()
            scores = self.classifier.classify_flood(
                sensor_data=ctx.sensor_data,
                detection_data=detections,
                image_size=self.params.image_size,
                timestamp=timestamp,
            )
            combined = scores.get("combined_score", 0.0)
            image_score = scores.get("image_score", 0.0)
            sensor_boost = scores.get("sensor_boost", 0.0)
            sensor_prediction = scores.get("sensor_prediction", "neutral")
            prediction = scores.get("final_prediction", 0)

            conflict = self._detect_conflict(combined, image_score, sensor_boost)
            drift = self._detect_drift(ctx.sensor_data)

            mid_band = self.params.threshold_low <= combined < self.params.threshold_high
            ambiguous = mid_band or conflict
            self._update_counters(combined, ambiguous, conflict, mid_band)
            self._score_history.append(combined)
            flapping = self._is_flapping()

            ctx.metadata["sensor_anomalies"] = scores.get("anomalies")
            classification_duration = perf_counter() - classification_phase_start
            classification_measure_duration = classification_duration

        fsm_timing["fsm_classification_core_s"] = classification_duration
        if energy_tracking_enabled and get_energy_tracker is not None:
            fsm_timing["fsm_energy_measurement_overhead_s"] = max(0.0, classification_measure_duration - classification_duration)

        llm_used = False
        llm_prediction: Optional[int] = None
        llm_backend_info: Dict[str, Any] = {}
        if (
            self.llm_enabled
            and self.dispatcher is not None
            and prev_state in (FloodState.S1, FloodState.S3)
            and motion == MotionState.STOP
            and self._counters["ambiguous"] >= self.params.frames_ambiguous
            and self._llm_last_used.get(prev_state, -1) < self._frame_index - self.params.frames_ambiguous
            and self.state != FloodState.S5
        ):
            llm_call_start = perf_counter()
            llm_call_duration: Optional[float] = None
            try:
                llm_result = self.dispatcher.run_remote_llm(
                    state=prev_state,
                    frame_index=self._frame_index,
                    ctx=ctx,
                )
                llm_call_duration = perf_counter() - llm_call_start
            except Exception as exc:
                llm_call_duration = perf_counter() - llm_call_start
                print(f"Remote LLM request failed ({exc}); continuing without LLM confirmation.")
                llm_result = None

            if llm_result:
                llm_prediction = llm_result.get("prediction")
                if llm_prediction is not None:
                    llm_prediction = int(llm_prediction)
                    llm_used = True
                    self._llm_last_used[prev_state] = self._frame_index
                    prediction = llm_prediction
                    llm_backend_info = {
                        "latency_s": llm_result.get("latency_s"),
                        "request_id": llm_result.get("request_id"),
                        "sent_at": llm_result.get("sent_at"),
                        "received_at": llm_result.get("received_at"),
                        "timing": llm_result.get("timing"),
                    }
                    if llm_call_duration is not None:
                        llm_backend_info["duration_s"] = llm_call_duration
                        fsm_timing["fsm_llm_request_s"] = llm_call_duration

        determine_start = perf_counter()
        next_state = self._determine_next_state(
            current_state=self.state,
            combined=combined,
            resource_flag=resource_flag,
            conflict=conflict,
            flapping=flapping,
        )
        fsm_timing["fsm_state_evaluation_s"] = perf_counter() - determine_start
        
        # Debug: State transition
        if next_state != self.state:
            print(f"🔄 [FSM] State transition: {self.state.value} → {next_state.value} (frame {self._frame_index}, "
                  f"counters={self._counters}, combined_score={combined:.3f})")
        else:
            print(f"🔍 [FSM] State unchanged: {self.state.value} (frame {self._frame_index}, "
                  f"counters={self._counters}, combined_score={combined:.3f})")

        enforce_start = perf_counter()
        (
            next_state,
            prediction,
            llm_used,
            llm_prediction,
            llm_backend_info,
        ) = self._enforce_s2_confirmation(
            prev_state=prev_state,
            candidate_state=next_state,
            resource_flag=resource_flag,
            ctx=ctx,
            motion=motion,
            prediction=prediction,
            llm_used=llm_used,
            llm_prediction=llm_prediction,
            llm_backend_info=llm_backend_info,
        )
        fsm_timing["fsm_s2_enforce_s"] = perf_counter() - enforce_start
        self.state = next_state
        if self.state != FloodState.S5:
            self._resource_anchor_state = None
            self._resource_skip_cursor = 0

        # Add FSM timing metrics
        fsm_total_duration = perf_counter() - fsm_start
        fsm_timing["fsm_total_s"] = fsm_total_duration
        timing_info.update(fsm_timing)
        
        timing_info["classification_s"] = classification_duration
        llm_duration_val = llm_backend_info.get("duration_s") if isinstance(llm_backend_info, dict) else None
        if isinstance(llm_duration_val, (int, float)):
            timing_info["llm_s"] = float(llm_duration_val)
        if isinstance(llm_backend_info, dict):
            llm_backend_timing = llm_backend_info.get("timing")
            if isinstance(llm_backend_timing, dict):
                for key, value in llm_backend_timing.items():
                    if isinstance(value, (int, float)):
                        timing_info[f"llm_backend_{key}"] = float(value)
        
        # Add FSM energy metrics to timing info
        if fsm_energy_metrics:
            if isinstance(fsm_energy_metrics.get("energy_j"), (int, float)):
                timing_info["fsm_energy_j"] = float(fsm_energy_metrics["energy_j"])
            if isinstance(fsm_energy_metrics.get("power_w"), (int, float)):
                timing_info["fsm_power_w"] = float(fsm_energy_metrics["power_w"])
            if isinstance(fsm_energy_metrics.get("cpu_util_%"), (int, float)):
                timing_info["fsm_cpu_util_%"] = float(fsm_energy_metrics["cpu_util_%"])

        finalize_start = perf_counter()
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
                "sensor_prediction": sensor_prediction,
            },
            counters=self._counters.copy(),
            llm_used=llm_used,
            llm_prediction=llm_prediction,
            conflict=conflict,
            drift=drift,
            flapping=flapping,
            skipped=False,
            model_switched=switched,
            backend=backend_name,
            backend_info=backend_info,
            llm_backend_info=llm_backend_info,
            s2_llm_confirmed=self._s2_llm_confirmed,
            timing=timing_info,
        )
        fsm_timing["fsm_finalize_s"] = perf_counter() - finalize_start
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
            scores={"combined_score": 0.0, "image_score": 0.0, "sensor_boost": 0.0, "sensor_prediction": "neutral"},
            counters=self._counters.copy(),
            llm_used=False,
            llm_prediction=None,
            conflict=False,
            drift=False,
            flapping=False,
            skipped=True,
            model_switched=False,
            tier_requested=self.model_manager.active_tier or ModelTier.NANO,
            backend="local",
            backend_info={},
            llm_backend_info={},
            s2_llm_confirmed=False,
            timing={},
        )

    def _enforce_s2_confirmation(
        self,
        *,
        prev_state: FloodState,
        candidate_state: FloodState,
        resource_flag: bool,
        ctx: FrameContext,
        motion: MotionState,
        prediction: int,
        llm_used: bool,
        llm_prediction: Optional[int],
        llm_backend_info: Dict[str, Any],
    ) -> Tuple[FloodState, int, bool, Optional[int], Dict[str, Any]]:
        next_state = candidate_state
        updated_prediction = prediction
        updated_llm_used = llm_used
        updated_llm_prediction = llm_prediction
        updated_backend = llm_backend_info

        if next_state == FloodState.S2 and prev_state != FloodState.S2:
            if not self.llm_enabled or self.dispatcher is None:
                self._s2_llm_confirmed = True
                return next_state, updated_prediction, updated_llm_used, updated_llm_prediction, updated_backend

            min_gap = max(1, self.params.frames_ambiguous)
            if self._s2_last_confirm_frame >= 0 and (self._frame_index - self._s2_last_confirm_frame) < min_gap:
                self._s2_llm_confirmed = False
                return (
                    prev_state,
                    updated_prediction,
                    updated_llm_used,
                    updated_llm_prediction,
                    updated_backend,
                )

            llm_result: Optional[Dict[str, Any]]
            llm_call_start = perf_counter()
            llm_call_duration: Optional[float] = None
            try:
                llm_result = self.dispatcher.run_remote_llm(
                    state=FloodState.S2,
                    frame_index=self._frame_index,
                    ctx=ctx,
                )
                llm_call_duration = perf_counter() - llm_call_start
            except Exception as exc:  # pragma: no cover - network/remote errors
                llm_call_duration = perf_counter() - llm_call_start
                print(f"Remote LLM confirmation failed ({exc}); postponing flood confirmation.")
                llm_result = None

            self._s2_last_confirm_frame = self._frame_index

            if not llm_result:
                self._s2_llm_confirmed = False
                return (
                    prev_state,
                    updated_prediction,
                    updated_llm_used,
                    updated_llm_prediction,
                    updated_backend,
                )

            prediction_raw = llm_result.get("prediction")
            try:
                confirm_prediction = int(prediction_raw) if prediction_raw is not None else None
            except (TypeError, ValueError):
                confirm_prediction = None

            updated_llm_used = True
            updated_llm_prediction = confirm_prediction
            updated_backend = {
                "latency_s": llm_result.get("latency_s"),
                "request_id": llm_result.get("request_id"),
                "sent_at": llm_result.get("sent_at"),
                "received_at": llm_result.get("received_at"),
                "timing": llm_result.get("timing"),
            }
            if llm_call_duration is not None:
                updated_backend["duration_s"] = llm_call_duration
            self._llm_last_used[FloodState.S2] = self._frame_index

            if confirm_prediction in (1, 2):
                self._s2_llm_confirmed = True
                updated_prediction = confirm_prediction
                return next_state, updated_prediction, updated_llm_used, updated_llm_prediction, updated_backend

            self._s2_llm_confirmed = False
            self._counters["high"] = 0
            return (
                prev_state,
                0,
                updated_llm_used,
                updated_llm_prediction,
                updated_backend,
            )

        if (
            next_state == FloodState.S2
            and prev_state == FloodState.S2
            and self.llm_enabled
            and self.dispatcher is not None
            and not resource_flag
            and motion != MotionState.FAST
            and self._llm_last_used.get(FloodState.S2, -1) < self._frame_index - self.params.frames_ambiguous
        ):
            llm_optional_start = perf_counter()
            llm_optional_duration: Optional[float] = None
            try:
                llm_result = self.dispatcher.run_remote_llm(
                    state=FloodState.S2,
                    frame_index=self._frame_index,
                    ctx=ctx,
                )
                llm_optional_duration = perf_counter() - llm_optional_start
            except Exception as exc:  # pragma: no cover - network/remote errors
                llm_optional_duration = perf_counter() - llm_optional_start
                print(f"Remote LLM request failed ({exc}); continuing without optional S2 confirmation.")
                llm_result = None
            if llm_result:
                optional_raw = llm_result.get("prediction")
                try:
                    optional_prediction = int(optional_raw) if optional_raw is not None else None
                except (TypeError, ValueError):
                    optional_prediction = None
                if optional_prediction is not None:
                    updated_llm_used = True
                    updated_llm_prediction = optional_prediction
                    updated_backend = {
                        "latency_s": llm_result.get("latency_s"),
                        "request_id": llm_result.get("request_id"),
                        "sent_at": llm_result.get("sent_at"),
                        "received_at": llm_result.get("received_at"),
                        "timing": llm_result.get("timing"),
                    }
                    if llm_optional_duration is not None:
                        updated_backend["duration_s"] = llm_optional_duration
                    self._llm_last_used[FloodState.S2] = self._frame_index
                    if optional_prediction in (1, 2):
                        updated_prediction = optional_prediction
                    elif optional_prediction == 0:
                        updated_prediction = 0

        if next_state != FloodState.S2:
            self._s2_llm_confirmed = False

        return next_state, updated_prediction, updated_llm_used, updated_llm_prediction, updated_backend

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
        backend: Optional[str] = None,
        backend_info: Optional[Dict[str, Any]] = None,
        llm_backend_info: Optional[Dict[str, Any]] = None,
        s2_llm_confirmed: bool = False,
        timing: Optional[Dict[str, float]] = None,
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
                backend=backend if backend is not None else base_decision.backend,
                backend_info=backend_info or base_decision.backend_info,
                llm_backend_info=llm_backend_info or base_decision.llm_backend_info,
                s2_llm_confirmed=s2_llm_confirmed,
                timing=timing or base_decision.timing,
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
            backend=backend or "local",
            backend_info=backend_info or {},
            llm_backend_info=llm_backend_info or {},
            s2_llm_confirmed=s2_llm_confirmed,
            timing=timing or {},
        )

    def _choose_tier(self, motion: MotionState, *, resource_flag: bool) -> ModelTier:
        # Check for fixed tier override (Ablation 6)
        import config
        fixed_tier = getattr(config, "FIXED_TIER", None)
        if fixed_tier:
            return ModelTier(fixed_tier)
        
        # Check for fast motion tier override (Ablation 5)
        if motion == MotionState.FAST:
            fast_tier = getattr(config, "FAST_MOTION_TIER", None)
            if fast_tier:
                return ModelTier(fast_tier)
        
        anchor_state = self._resource_anchor_state if self.state == FloodState.S5 else self.state
        tiers = [ModelTier.NANO, ModelTier.SMALL, ModelTier.MEDIUM, ModelTier.LARGE]

        if anchor_state == FloodState.S0:
            base_tier = ModelTier.NANO
        elif anchor_state == FloodState.S1:
            if motion == MotionState.FAST:
                base_tier = ModelTier.SMALL
            else:
                base_tier = ModelTier.MEDIUM
        elif anchor_state == FloodState.S2:
            if motion == MotionState.FAST:
                base_tier = ModelTier.SMALL
            else:
                base_tier = ModelTier.LARGE
        elif anchor_state == FloodState.S3:
            current = self.model_manager.active_tier or ModelTier.MEDIUM
            if tiers.index(current) >= tiers.index(ModelTier.MEDIUM):
                base_tier = current.higher()
            else:
                base_tier = ModelTier.MEDIUM
        else:
            active = self.model_manager.active_tier
            base_tier = active if active is not None else ModelTier.NANO

        if self.state == FloodState.S5:
            # Default to nano when resource constrained
            degraded = ModelTier.NANO
            
            # Safety cap: ensure we never exceed SMALL when resource constrained
            if tiers.index(degraded) > tiers.index(ModelTier.SMALL):
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

    def _update_counters(
        self,
        combined: float,
        ambiguous: bool,
        conflict: bool,
        mid_band: bool,
    ) -> None:
        # Determine current frame's classification state
        current_is_high = combined >= self.params.threshold_high
        current_is_suspicious = current_is_high or ambiguous  # sus, ambiguous, or flood
        
        # Handle high counter
        if combined >= self.params.threshold_high:
            # If previous was suspicious (high/ambiguous) and current is also high, count as 2
            if self._prev_was_suspicious:
                self._counters["high"] += 2
            else:
                self._counters["high"] += 1
        else:
            self._counters["high"] = 0

        if combined < self.params.threshold_low:
            self._counters["low"] += 1
        else:
            self._counters["low"] = 0

        # Handle ambiguous counter
        if ambiguous:
            # If previous was suspicious (high/ambiguous) and current is also ambiguous, count as 2
            if self._prev_was_suspicious:
                self._counters["ambiguous"] += 2
            else:
                self._counters["ambiguous"] += 1
        else:
            self._counters["ambiguous"] = 0

        if conflict:
            self._counters["conflict"] += 1
        else:
            self._counters["conflict"] = 0

        if mid_band:
            self._counters["mid"] += 1
        else:
            self._counters["mid"] = 0
        
        # Update previous state for next frame
        self._prev_was_suspicious = current_is_suspicious

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
        conflict: bool,
        flapping: bool,
    ) -> FloodState:
        if resource_flag:
            return FloodState.S5

        low_stable = self._counters["low"] >= self.params.frames_low
        high_stable = self._counters["high"] >= self.params.frames_high
        mid_stable = self._counters["mid"] >= self.params.frames_high
        ambiguity_persist = self._counters["ambiguous"] >= self.params.frames_ambiguous
        conflict_persist = self._counters["conflict"] >= self.params.frames_ambiguous

        if current_state == FloodState.S5:
            if combined >= self.params.threshold_high:
                return FloodState.S2
            if combined < self.params.threshold_low:
                return FloodState.S0
            if ambiguity_persist or conflict_persist:
                return FloodState.S3
            return FloodState.S1

        if current_state == FloodState.S0:
            if conflict_persist:
                return FloodState.S3
            if high_stable:
                return FloodState.S2
            if mid_stable or ambiguity_persist:
                return FloodState.S1
            return FloodState.S0

        if current_state == FloodState.S1:
            if high_stable:
                return FloodState.S2
            if low_stable:
                return FloodState.S0
            if ambiguity_persist or conflict_persist:
                return FloodState.S3
            return FloodState.S1

        if current_state == FloodState.S2:
            if low_stable:
                return FloodState.S0
            if ambiguity_persist or conflict_persist:
                return FloodState.S3
            if mid_stable or flapping:
                return FloodState.S1
            return FloodState.S2

        if current_state == FloodState.S3:
            if high_stable:
                return FloodState.S2
            if low_stable:
                return FloodState.S0
            if not conflict and mid_stable:
                return FloodState.S1
            return FloodState.S3

        # Fallback - treat any unexpected state as S0.
        if high_stable:
            return FloodState.S2
        if mid_stable or ambiguity_persist:
            return FloodState.S1
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
        "backend": decision.backend,
        "backend_info": decision.backend_info,
        "llm_backend_info": decision.llm_backend_info,
        "s2_llm_confirmed": decision.s2_llm_confirmed,
        "timing": decision.timing,
    }
