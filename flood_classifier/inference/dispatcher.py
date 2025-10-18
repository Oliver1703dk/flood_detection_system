"""Inference dispatching utilities used by the Pi orchestrator.

The dispatcher decides whether to run a frame locally using the Pi's nano model
or to offload the work to the Jetson via MQTT.  It also exposes helpers used by
the remote LLM adapter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import config
from flood_classifier.inference.mqtt_backend import (
    MQTTInferenceClient,
    MQTTResponse,
    encode_image_for_transport,
)

if TYPE_CHECKING:  # pragma: no cover
    from flood_classifier.fsm.flood_fsm import FloodState, ModelTier


class RemoteInferenceError(RuntimeError):
    """Raised when the Jetson returns an error payload."""


@dataclass
class DispatcherResult:
    """Outcome of an inference request."""

    detections: List[Dict[str, object]]
    tier: "ModelTier"
    backend: str
    switched: bool
    metadata: Dict[str, object] = field(default_factory=dict)


class MQTTJetsonBackend:
    """Wrapper around :class:`MQTTInferenceClient` with task-specific helpers."""

    def __init__(self, client: Optional[MQTTInferenceClient] = None) -> None:
        self._client = client or MQTTInferenceClient()

    def is_healthy(self) -> bool:
        return self._client.is_healthy()

    def _run_task(self, task: str, payload: Dict[str, object]) -> MQTTResponse:
        envelope = {"task": task, **payload}
        response = self._client.publish_request(envelope)
        data = response.data
        if data.get("error"):
            raise RemoteInferenceError(str(data["error"]))
        return response

    def run_yolo(
        self,
        *,
        image_b64: str,
        tier: "ModelTier",
        frame_index: int,
        fsm_state: "FloodState",
        metadata: Dict[str, object],
        sensor_data: Optional[Dict[str, Any]] = None,
        sensor_baseline: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, object]:
        payload = {
            "tier": tier.value,
            "frame_index": frame_index,
            "state": fsm_state.value,
            "image_b64": image_b64,
            "metadata": metadata,
        }
        if sensor_data is not None:
            payload["sensor_data"] = sensor_data
        if sensor_baseline is not None:
            payload["sensor_baseline"] = sensor_baseline
        response = self._run_task("yolo", payload)
        data = response.data
        return {
            "detections": data.get("detections", []),
            "latency_s": response.latency_s,
            "request_id": data.get("id"),
            "sent_at": response.sent_at,
            "received_at": response.received_at,
            "timing": data.get("timing"),
        }

    def run_llm(
        self,
        *,
        image_b64: str,
        frame_index: int,
        fsm_state: "FloodState",
        metadata: Dict[str, object],
        sensor_data: Optional[Dict[str, Any]] = None,
        sensor_baseline: Optional[Dict[str, Any]] = None,
        sensor_anomalies: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, object]:
        payload = {
            "frame_index": frame_index,
            "state": fsm_state.value,
            "image_b64": image_b64,
            "metadata": metadata,
        }
        if sensor_data is not None:
            payload["sensor_data"] = sensor_data
        if sensor_baseline is not None:
            payload["sensor_baseline"] = sensor_baseline
        if sensor_anomalies is not None:
            payload["sensor_anomalies"] = sensor_anomalies
        response = self._run_task("llm", payload)
        data = response.data
        return {
            "prediction": data.get("prediction"),
            "latency_s": response.latency_s,
            "request_id": data.get("id"),
            "sent_at": response.sent_at,
            "received_at": response.received_at,
            "timing": data.get("timing"),
        }


class InferenceDispatcher:
    """Decide where to run inference and proxy the actual execution."""

    def __init__(
        self,
        remote_backend: Optional[MQTTJetsonBackend] = None,
        routing: Optional[Dict[str, str]] = None,
        local_tier: str = config.LOCAL_YOLO_TIER,
    ) -> None:
        self.remote_backend = remote_backend
        self.routing = routing or config.INFERENCE_ROUTING
        self.local_tier = local_tier

    def _route_for_state(self, state: "FloodState") -> str:
        return self.routing.get(state.value, self.routing.get("default", "local"))

    def should_route_remote(
        self,
        *,
        state: "FloodState",
        requested_tier: "ModelTier",
        llm_required: bool = False,
    ) -> bool:
        if self.remote_backend is None:
            return False

        if llm_required:
            return True

        if requested_tier.value != self.local_tier:
            return True

        return self._route_for_state(state) == "remote"

    def try_remote_yolo(
        self,
        *,
        state: "FloodState",
        requested_tier: "ModelTier",
        frame_index: int,
        ctx,
    ) -> Optional[DispatcherResult]:
        if not self.should_route_remote(state=state, requested_tier=requested_tier):
            return None

        if self.remote_backend is None or not self.remote_backend.is_healthy():
            return None

        image_b64 = getattr(ctx, "image_b64", None)
        if not image_b64:
            image_bytes = ctx.image_bytes()
            if image_bytes is None:
                raise ValueError("FrameContext must provide image data for remote inference")
            image_b64 = encode_image_for_transport(image_bytes)

        metadata = {
            "timestamp": ctx.metadata.get("timestamp"),
            "camera_id": ctx.metadata.get("camera_id"),
            "video_file": ctx.metadata.get("video_file"),
            "video_timestamp_sec": ctx.metadata.get("video_timestamp_sec"),
            "sensor_data": ctx.sensor_data,
            "sensor_baseline": ctx.metadata.get("sensor_baseline"),
        }

        result = self.remote_backend.run_yolo(
            image_b64=image_b64,
            tier=requested_tier,
            frame_index=frame_index,
            fsm_state=state,
            metadata=metadata,
            sensor_data=ctx.sensor_data,
            sensor_baseline=ctx.metadata.get("sensor_baseline"),
        )
        return DispatcherResult(
            detections=result.get("detections", []),
            tier=requested_tier,
            backend="remote",
            switched=False,
            metadata={
                "latency_s": result.get("latency_s"),
                "request_id": result.get("request_id"),
                "sent_at": result.get("sent_at"),
                "received_at": result.get("received_at"),
                "timing": result.get("timing"),
            },
        )

    def run_remote_llm(
        self,
        *,
        state: "FloodState",
        frame_index: int,
        ctx,
    ) -> Optional[Dict[str, object]]:
        if self.remote_backend is None:
            return None

        if not self.remote_backend.is_healthy():
            return None

        image_bytes = ctx.image_bytes()
        if image_bytes is None:
            return None

        metadata = {
            "timestamp": ctx.metadata.get("timestamp"),
            "camera_id": ctx.metadata.get("camera_id"),
            "video_file": ctx.metadata.get("video_file"),
            "video_timestamp_sec": ctx.metadata.get("video_timestamp_sec"),
        }
        result = self.remote_backend.run_llm(
            image_b64=encode_image_for_transport(image_bytes),
            frame_index=frame_index,
            fsm_state=state,
            metadata=metadata,
            sensor_data=ctx.sensor_data,
            sensor_baseline=ctx.metadata.get("sensor_baseline"),
            sensor_anomalies=ctx.metadata.get("sensor_anomalies"),
        )
        return result
