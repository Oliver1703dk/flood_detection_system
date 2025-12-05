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

    def infer(self, ctx, requested_tier: ModelTier, frame_index: int, fsm_state):
        switched = False
        if self._active_tier is None or requested_tier != self._active_tier:
            switched = True
            self._active_tier = requested_tier
        self.calls.append(requested_tier)
        return [], self._active_tier, switched, {"backend": "local", "metadata": {}}

    def prewarm_models(self, tiers=None):
        """No-op prewarm for tests."""
        return {}


class CooldownModelManager(DummyModelManager):
    def __init__(self, cooldown: int):
        super().__init__()
        self.cooldown = cooldown
        self.last_switch = -cooldown

    def infer(self, ctx, requested_tier: ModelTier, frame_index: int, fsm_state):
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
        return [], self._active_tier, switched, {"backend": "local", "metadata": {}}


class RecordingModelManager(DummyModelManager):
    def infer(self, ctx, requested_tier: ModelTier, frame_index: int, fsm_state):
        detections, tier, switched, meta = super().infer(ctx, requested_tier, frame_index, fsm_state)
        return detections, tier, switched, meta


class StubDispatcher:
    def __init__(self, llm_response: Optional[int] = None):
        self.llm_response = llm_response
        self.llm_calls = 0

    def try_remote_yolo(self, **_):
        return None

    def run_remote_llm(self, **_):
        if self.llm_response is None:
            return None
        self.llm_calls += 1
        return {"prediction": self.llm_response, "latency_s": 0.01, "request_id": "stub"}


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
    fsm = FloodFSM(params=params, classifier=classifier, model_manager=model_manager, dispatcher=StubDispatcher())

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


def test_ambiguous_state_escalates_to_large_tier():
    params = FSMParams(
        frames_high=2,
        frames_low=2,
        frames_ambiguous=1,
        accuracy_weight=0.8,
        timeliness_weight=0.1,
        energy_weight=0.1,
    )
    classifier = DummyClassifier(
        [
            {"final_prediction": 1, "combined_score": 0.45, "image_score": 0.45, "sensor_boost": 0.05},
            {"final_prediction": 1, "combined_score": 0.5, "image_score": 0.2, "sensor_boost": 0.3},
            {"final_prediction": 1, "combined_score": 0.5, "image_score": 0.45, "sensor_boost": 0.05},
        ]
    )
    model_manager = DummyModelManager()
    fsm = FloodFSM(params=params, classifier=classifier, model_manager=model_manager, dispatcher=StubDispatcher())

    decision1 = fsm.next_state(make_context(motion="slow"))
    assert decision1.state == FloodState.S1
    assert decision1.model_tier == ModelTier.NANO

    decision2 = fsm.next_state(make_context(motion="stop"))
    assert decision2.state == FloodState.S3
    assert decision2.model_tier == ModelTier.MEDIUM

    decision3 = fsm.next_state(make_context(motion="stop"))
    assert decision3.tier_requested == ModelTier.LARGE
    assert decision3.model_tier == ModelTier.LARGE


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
    fsm = FloodFSM(params=params, classifier=classifier, model_manager=model_manager, dispatcher=StubDispatcher())

    for _ in range(5):
        fsm.next_state(make_context(motion="stop"))

    last_decision = fsm.next_state(make_context())
    assert last_decision.model_tier == ModelTier.MEDIUM  # still medium due to cooldown lock
    assert not last_decision.model_switched
    assert model_manager.calls.count(ModelTier.MEDIUM) >= 1


def test_llm_only_triggers_in_stop_ambiguous_states():
    params = FSMParams(frames_ambiguous=1, llm_enabled=True)
    classifier = DummyClassifier(
        [
            {"final_prediction": 1, "combined_score": 0.5, "image_score": 0.45, "sensor_boost": 0.05},
            {"final_prediction": 1, "combined_score": 0.5, "image_score": 0.45, "sensor_boost": 0.05},
            {"final_prediction": 2, "combined_score": 0.7, "image_score": 0.65, "sensor_boost": 0.05},
        ]
    )
    model_manager = DummyModelManager()
    dispatcher = StubDispatcher(llm_response=2)
    fsm = FloodFSM(params=params, classifier=classifier, model_manager=model_manager, dispatcher=dispatcher)

    first = fsm.next_state(make_context(motion="slow"))
    second = fsm.next_state(make_context(motion="stop"))
    third = fsm.next_state(make_context(motion="stop"))

    assert not first.llm_used
    assert second.llm_used and second.llm_prediction == 2
    assert dispatcher.llm_calls == 1
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
    fsm = FloodFSM(params=params, classifier=classifier, model_manager=model_manager, dispatcher=StubDispatcher())

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

    skipped_again = fsm.next_state(make_context(motion="stop", resource=True))
    assert skipped_again.skipped
    assert skipped_again.state == FloodState.S5
    assert len(model_manager.calls) == calls_after_first  # still no inference

    resumed = fsm.next_state(make_context(motion="stop", resource=True))
    assert not resumed.skipped
    assert resumed.state == FloodState.S5
    assert len(model_manager.calls) == calls_after_first + 1


def test_resource_constrained_uses_nano_for_non_s2_states():
    """Verify that resource constrained mode uses NANO for S0, S1, S3 anchor states."""
    params = FSMParams(frames_high=2, frames_ambiguous=2)
    # Use low score to stay in S0
    classifier_s0 = DummyClassifier([
        {"final_prediction": 0, "combined_score": 0.3, "image_score": 0.25, "sensor_boost": 0.05},
    ])
    model_manager = RecordingModelManager()
    fsm = FloodFSM(params=params, classifier=classifier_s0, model_manager=model_manager, dispatcher=StubDispatcher())
    
    # Test S0 -> S5: should use NANO
    decision_s0 = fsm.next_state(make_context(motion="stop", resource=True))
    assert decision_s0.state == FloodState.S5
    assert decision_s0.model_tier == ModelTier.NANO
    assert decision_s0.tier_requested == ModelTier.NANO
    
    # Test S1 -> S5: should use NANO
    # Use ambiguous scores (between threshold_low 0.12 and threshold_high 0.4) to reach S1
    classifier_s1 = DummyClassifier([
        {"final_prediction": 1, "combined_score": 0.5, "image_score": 0.45, "sensor_boost": 0.05},
        {"final_prediction": 1, "combined_score": 0.5, "image_score": 0.45, "sensor_boost": 0.05},
        {"final_prediction": 1, "combined_score": 0.5, "image_score": 0.45, "sensor_boost": 0.05},
    ])
    model_manager_s1 = RecordingModelManager()
    fsm_s1 = FloodFSM(params=params, classifier=classifier_s1, model_manager=model_manager_s1, dispatcher=StubDispatcher())
    # Reach S1 first (need frames_ambiguous=2 ambiguous frames)
    fsm_s1.next_state(make_context(motion="slow"))
    fsm_s1.next_state(make_context(motion="slow"))
    # Now we should be in S1, enter resource constrained mode
    decision_s1 = fsm_s1.next_state(make_context(motion="slow", resource=True))
    assert decision_s1.state == FloodState.S5
    assert decision_s1.model_tier == ModelTier.NANO
    assert decision_s1.tier_requested == ModelTier.NANO


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
    fsm = FloodFSM(params=params, classifier=classifier, model_manager=model_manager, dispatcher=StubDispatcher())

    fsm.next_state(make_context())
    fsm.next_state(make_context())
    fsm.next_state(make_context())
    fsm.next_state(make_context())
    fsm.next_state(make_context())
    decision = fsm.next_state(make_context(sensor_data={"drift": True}))

    assert decision.state == FloodState.S3
    assert decision.flapping
    assert decision.drift
