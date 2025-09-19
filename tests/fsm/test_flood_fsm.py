from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, List, Optional

from flood_classifier.fsm.flood_fsm import (
    FSMParams,
    FloodFSM,
    FloodState,
    FrameContext,
    ModelTier,
)


class DummyClassifier:
    """Deterministic classifier returning predefined score dictionaries."""

    def __init__(self, outputs: Iterable[Dict[str, float]]):
        self._sequence = list(outputs)
        self._idx = 0

    def classify_flood(self, **_):
        if self._idx >= len(self._sequence):
            return self._sequence[-1]
        result = self._sequence[self._idx]
        self._idx += 1
        return result


class DummyModelManager:
    """Bare-bones model manager that records tier requests."""

    def __init__(self):
        self._active_tier: Optional[ModelTier] = None
        self.calls: List[ModelTier] = []

    @property
    def active_tier(self) -> Optional[ModelTier]:
        return self._active_tier

    def infer(self, ctx, requested_tier: ModelTier, frame_index: int):
        switched = False
        if self._active_tier is None or requested_tier != self._active_tier:
            switched = True
            self._active_tier = requested_tier
        self.calls.append(requested_tier)
        return [], self._active_tier, switched


class CooldownModelManager(DummyModelManager):
    def __init__(self, cooldown: int):
        super().__init__()
        self.cooldown = cooldown
        self.last_switch = -cooldown

    def infer(self, ctx, requested_tier: ModelTier, frame_index: int):
        switched = False
        if self._active_tier is None:
            self._active_tier = requested_tier
            self.last_switch = frame_index
            switched = True
        elif requested_tier != self._active_tier:
            if frame_index - self.last_switch >= self.cooldown:
                self._active_tier = requested_tier
                self.last_switch = frame_index
                switched = True
            requested_tier = self._active_tier
        self.calls.append(requested_tier)
        return [], self._active_tier, switched


class RecordingModelManager(DummyModelManager):
    def infer(self, ctx, requested_tier: ModelTier, frame_index: int):
        detections, tier, switched = super().infer(ctx, requested_tier, frame_index)
        return detections, tier, switched


class DummyLLM:
    def __init__(self, response: int = 1):
        self.calls = 0
        self.response = response

    def classify_flood(self, _image_bytes: bytes) -> int:
        self.calls += 1
        return self.response


def make_context(
    *,
    motion: Optional[str] = None,
    resource: bool = False,
    sensor_data: Optional[Dict[str, float]] = None,
) -> FrameContext:
    return FrameContext(
        image_b64=None,
        sensor_data=sensor_data or {},
        timestamp=datetime.utcnow(),
        motion_hint=motion,
        resource_constrained=resource,
        metadata={},
    )


def test_fsm_state_progression_and_tier_selection():
    params = FSMParams(
        frames_high=2,
        frames_low=2,
        frames_ambiguous=1,
        accuracy_weight=0.7,
        timeliness_weight=0.2,
        energy_weight=0.1,
    )
    classifier = DummyClassifier(
        [
            {"final_prediction": 0, "combined_score": 0.2, "image_score": 0.2, "sensor_boost": 0.0},
            {"final_prediction": 1, "combined_score": 0.5, "image_score": 0.45, "sensor_boost": 0.05},
            {"final_prediction": 2, "combined_score": 0.7, "image_score": 0.65, "sensor_boost": 0.05},
            {"final_prediction": 2, "combined_score": 0.75, "image_score": 0.7, "sensor_boost": 0.05},
        ]
    )
    model_manager = DummyModelManager()
    fsm = FloodFSM(params=params, classifier=classifier, model_manager=model_manager, llm_classifier=None)

    decisions = [
        fsm.next_state(make_context()),
        fsm.next_state(make_context()),
        fsm.next_state(make_context(motion="stop")),
        fsm.next_state(make_context(motion="stop")),
    ]

    assert decisions[0].state == FloodState.S0
    assert decisions[0].model_tier == ModelTier.NANO
    assert decisions[1].state == FloodState.S1
    assert decisions[2].model_tier == ModelTier.MEDIUM  # accuracy-heavy S1 prefers medium on slow/stop
    assert decisions[3].state == FloodState.S2


def test_model_cooldown_enforced():
    params = FSMParams(
        frames_high=2,
        frames_low=1,
        frames_ambiguous=1,
        model_cooldown=3,
        accuracy_weight=0.7,
    )
    classifier = DummyClassifier(
        [
            {"final_prediction": 0, "combined_score": 0.2, "image_score": 0.2, "sensor_boost": 0.0},
            {"final_prediction": 0, "combined_score": 0.2, "image_score": 0.2, "sensor_boost": 0.0},
            {"final_prediction": 0, "combined_score": 0.2, "image_score": 0.2, "sensor_boost": 0.0},
            {"final_prediction": 1, "combined_score": 0.5, "image_score": 0.45, "sensor_boost": 0.05},
            {"final_prediction": 1, "combined_score": 0.5, "image_score": 0.45, "sensor_boost": 0.05},
            {"final_prediction": 0, "combined_score": 0.2, "image_score": 0.2, "sensor_boost": 0.0},
        ]
    )
    model_manager = CooldownModelManager(cooldown=params.model_cooldown)
    fsm = FloodFSM(params=params, classifier=classifier, model_manager=model_manager, llm_classifier=None)

    for _ in range(5):
        fsm.next_state(make_context(motion="stop"))

    last_decision = fsm.next_state(make_context())
    assert last_decision.model_tier == ModelTier.MEDIUM  # still medium due to cooldown lock
    assert not last_decision.model_switched
    assert model_manager.calls.count(ModelTier.MEDIUM) >= 1


def test_llm_only_triggers_in_stop_ambiguous_states():
    params = FSMParams(frames_ambiguous=2)
    classifier = DummyClassifier(
        [
            {"final_prediction": 1, "combined_score": 0.5, "image_score": 0.45, "sensor_boost": 0.05},
            {"final_prediction": 1, "combined_score": 0.5, "image_score": 0.45, "sensor_boost": 0.05},
            {"final_prediction": 2, "combined_score": 0.7, "image_score": 0.65, "sensor_boost": 0.05},
        ]
    )
    model_manager = DummyModelManager()
    llm = DummyLLM(response=2)
    fsm = FloodFSM(params=params, classifier=classifier, model_manager=model_manager, llm_classifier=llm)

    first = fsm.next_state(make_context(motion="slow"))
    second = fsm.next_state(make_context(motion="stop"))
    third = fsm.next_state(make_context(motion="stop"))

    assert not first.llm_used
    assert second.llm_used and second.llm_prediction == 2
    assert llm.calls == 1
    assert not third.llm_used  # confirmation limited to S1/S3


def test_resource_state_drops_tier_and_skips_frames():
    params = FSMParams(frames_high=2)
    classifier = DummyClassifier(
        [
            {"final_prediction": 1, "combined_score": 0.5, "image_score": 0.45, "sensor_boost": 0.05},
            {"final_prediction": 2, "combined_score": 0.7, "image_score": 0.65, "sensor_boost": 0.05},
            {"final_prediction": 2, "combined_score": 0.72, "image_score": 0.68, "sensor_boost": 0.05},
            {"final_prediction": 2, "combined_score": 0.74, "image_score": 0.69, "sensor_boost": 0.05},
        ]
    )
    model_manager = RecordingModelManager()
    fsm = FloodFSM(params=params, classifier=classifier, model_manager=model_manager, llm_classifier=None)

    # Reach S2 first (two high frames)
    fsm.next_state(make_context(motion="stop"))
    fsm.next_state(make_context(motion="stop"))
    fsm.next_state(make_context(motion="stop"))

    # Enter resource constrained mode
    decision_resource = fsm.next_state(make_context(motion="stop", resource=True))
    assert decision_resource.state == FloodState.S5
    assert decision_resource.model_tier == ModelTier.SMALL
    assert decision_resource.tier_requested == ModelTier.SMALL
    calls_after_first = len(model_manager.calls)

    skipped = fsm.next_state(make_context(motion="stop", resource=True))
    assert skipped.skipped
    assert skipped.state == FloodState.S5
    assert len(model_manager.calls) == calls_after_first  # no inference during skip

    resumed = fsm.next_state(make_context(motion="stop", resource=True))
    assert not resumed.skipped
    assert resumed.state == FloodState.S5
    assert len(model_manager.calls) == calls_after_first + 1


def test_flapping_moves_to_s3_and_detects_drift():
    params = FSMParams(frames_high=2, frames_low=2)
    classifier = DummyClassifier(
        [
            {"final_prediction": 1, "combined_score": 0.5, "image_score": 0.45, "sensor_boost": 0.05},
            {"final_prediction": 2, "combined_score": 0.7, "image_score": 0.65, "sensor_boost": 0.05},
            {"final_prediction": 2, "combined_score": 0.72, "image_score": 0.68, "sensor_boost": 0.05},
            {"final_prediction": 0, "combined_score": 0.2, "image_score": 0.22, "sensor_boost": 0.0},
            {"final_prediction": 2, "combined_score": 0.74, "image_score": 0.69, "sensor_boost": 0.05},
            {"final_prediction": 0, "combined_score": 0.2, "image_score": 0.21, "sensor_boost": 0.0},
        ]
    )
    model_manager = DummyModelManager()
    fsm = FloodFSM(params=params, classifier=classifier, model_manager=model_manager, llm_classifier=None)

    fsm.next_state(make_context())
    fsm.next_state(make_context())
    fsm.next_state(make_context())
    fsm.next_state(make_context())
    fsm.next_state(make_context())
    decision = fsm.next_state(make_context(sensor_data={"drift": True}))

    assert decision.state == FloodState.S3
    assert decision.flapping
    assert decision.drift
