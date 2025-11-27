"""Inference dispatching utilities used by the Pi orchestrator.

The dispatcher decides whether to run a frame locally using the Pi's nano model
or to offload the work to the Jetson via MQTT.  It also exposes helpers used by
the remote LLM adapter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import config
from flood_classifier.inference.mqtt_backend import (
    MQTTInferenceClient,
    MQTTResponse,
    encode_image_for_transport,
)

if TYPE_CHECKING:  # pragma: no cover
    from flood_classifier.fsm.flood_fsm import FloodState, ModelTier, MotionState


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

    def _run_task(self, task: str, payload: Dict[str, object], timing_dict: Optional[Dict[str, float]] = None) -> MQTTResponse:
        envelope = {"task": task, **payload}
        response = self._client.publish_request(envelope, timing_dict=timing_dict)
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
        timing_dict: Optional[Dict[str, float]] = None,
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
        response = self._run_task("yolo", payload, timing_dict=timing_dict)
        data = response.data
        result = {
            "detections": data.get("detections", []),
            "latency_s": response.latency_s,
            "request_id": data.get("id"),
            "sent_at": response.sent_at,
            "received_at": response.received_at,
            "timing": data.get("timing"),
        }
        # Merge timing_dict into result timing if available
        if timing_dict and result.get("timing"):
            result["timing"].update(timing_dict)
        elif timing_dict:
            result["timing"] = timing_dict
        return result

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
        timing_dict: Optional[Dict[str, float]] = None,
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
        response = self._run_task("llm", payload, timing_dict=timing_dict)
        data = response.data
        result = {
            "prediction": data.get("prediction"),
            "latency_s": response.latency_s,
            "request_id": data.get("id"),
            "sent_at": response.sent_at,
            "received_at": response.received_at,
            "timing": data.get("timing"),
        }
        # Merge timing_dict into result timing if available
        if timing_dict and result.get("timing"):
            result["timing"].update(timing_dict)
        elif timing_dict:
            result["timing"] = timing_dict
        return result


class InferenceDispatcher:
    """Decide where to run inference and proxy the actual execution."""

    def __init__(
        self,
        remote_backend: Optional[MQTTJetsonBackend] = None,
        routing: Optional[Dict[str, str]] = None,
        local_tier: str = config.LOCAL_YOLO_TIER,
        available_local_tiers: Optional[set] = None,
    ) -> None:
        self.remote_backend = remote_backend
        self.routing = routing or config.INFERENCE_ROUTING
        self.local_tier = local_tier
        # Set of tier values (strings) that are available locally
        # Defaults to just the local_tier if not provided
        self.available_local_tiers = available_local_tiers or {local_tier}

    def _route_for_state(self, state: "FloodState") -> str:
        return self.routing.get(state.value, self.routing.get("default", "local"))

    def should_route_remote(
        self,
        *,
        state: "FloodState",
        requested_tier: "ModelTier",
        llm_required: bool = False,
        motion: Optional["MotionState"] = None,
    ) -> bool:
        if self.remote_backend is None:
            print(f"🔍 [Dispatcher] should_route_remote: remote_backend is None, returning False")
            return False

        # Check for force offload override (Ablation 6)
        if getattr(config, "FORCE_OFFLOAD", False):
            print(f"🔍 [Dispatcher] should_route_remote: FORCE_OFFLOAD=True, routing to remote")
            return True

        # Check for fast motion offload if enabled
        if motion is not None:
            from flood_classifier.fsm.flood_fsm import MotionState
            if motion == MotionState.FAST and getattr(config, "ALWAYS_OFFLOAD_ON_FAST_MOTION", False):
                print(f"🔍 [Dispatcher] should_route_remote: motion=FAST and ALWAYS_OFFLOAD_ON_FAST_MOTION=True, routing to remote")
                return True

        if llm_required:
            print(f"🔍 [Dispatcher] should_route_remote: llm_required=True, routing to remote")
            return True

        # Route to remote if tier is not available locally
        if requested_tier.value not in self.available_local_tiers:
            print(f"🔍 [Dispatcher] should_route_remote: tier {requested_tier.value} not in available_local_tiers {self.available_local_tiers}, routing to remote")
            return True

        route_decision = self._route_for_state(state)
        should_route = route_decision == "remote"
        print(f"🔍 [Dispatcher] should_route_remote: state={state.value}, tier={requested_tier.value}, "
              f"available_local_tiers={self.available_local_tiers}, route_decision={route_decision}, "
              f"should_route={should_route}")
        return should_route

    def try_remote_yolo(
        self,
        *,
        state: "FloodState",
        requested_tier: "ModelTier",
        frame_index: int,
        ctx,
        motion: Optional["MotionState"] = None,
    ) -> Optional[DispatcherResult]:
        # Extract motion from ctx if not provided and ctx has get_motion_state method
        if motion is None and hasattr(ctx, "get_motion_state"):
            motion = ctx.get_motion_state()
        
        if not self.should_route_remote(state=state, requested_tier=requested_tier, motion=motion):
            print(f"🔍 [Dispatcher] try_remote_yolo: should_route_remote returned False, not routing to remote")
            return None

        if self.remote_backend is None:
            print(f"🔍 [Dispatcher] try_remote_yolo: remote_backend is None, cannot route to remote")
            return None
        
        is_healthy = self.remote_backend.is_healthy()
        if not is_healthy:
            print(f"🔍 [Dispatcher] try_remote_yolo: remote_backend is not healthy (is_healthy()={is_healthy}), cannot route to remote")
            return None
        
        print(f"🔍 [Dispatcher] try_remote_yolo: Routing to remote - state={state.value}, tier={requested_tier.value}, frame={frame_index}, backend_healthy={is_healthy}")

        # Debug: Sending request
        print(f"📤 [PI] Sending YOLO request to Jetson: state={state.value}, tier={requested_tier.value}, frame={frame_index}")
        send_start = perf_counter()

        # Create timing dictionary to collect all timing metrics
        timing_dict: Dict[str, float] = {}

        # Time image bytes retrieval
        image_bytes_start = perf_counter()
        image_b64 = getattr(ctx, "image_b64", None)
        if not image_b64:
            image_bytes = ctx.image_bytes()
            if image_bytes is None:
                raise ValueError("FrameContext must provide image data for remote inference")
            image_bytes_duration = perf_counter() - image_bytes_start
            timing_dict["image_bytes_retrieve_s"] = image_bytes_duration
            
            # Time image encoding
            image_b64 = encode_image_for_transport(image_bytes, timing_dict=timing_dict)
        else:
            image_bytes_duration = perf_counter() - image_bytes_start
            timing_dict["image_bytes_retrieve_s"] = image_bytes_duration

        # Debug: Image encoding
        encode_time = perf_counter() - send_start
        print(f"📦 [PI] Image encoded in {encode_time*1000:.1f}ms, size={len(image_b64)} bytes")

        metadata = {
            "timestamp": ctx.metadata.get("timestamp"),
            "camera_id": ctx.metadata.get("camera_id"),
            "video_file": ctx.metadata.get("video_file"),
            "video_timestamp_sec": ctx.metadata.get("video_timestamp_sec"),
            "sensor_data": ctx.sensor_data,
            "sensor_baseline": ctx.metadata.get("sensor_baseline"),
        }

        # Debug: Publishing request
        print(f"🚀 [PI] Publishing MQTT request to Jetson...")
        request_start = perf_counter()

        result = self.remote_backend.run_yolo(
            image_b64=image_b64,
            tier=requested_tier,
            frame_index=frame_index,
            fsm_state=state,
            metadata=metadata,
            sensor_data=ctx.sensor_data,
            sensor_baseline=ctx.metadata.get("sensor_baseline"),
            timing_dict=timing_dict,
        )
        
        # Debug: Response received
        total_time = perf_counter() - request_start
        latency = result.get("latency_s", 0)
        compute_time = result.get("timing", {}).get("compute_s", 0)
        network_time = timing_dict.get("network_pi_to_jetson_s", 0)
        print(f"📥 [PI] Received Jetson response in {total_time*1000:.1f}ms")
        print(f"   Breakdown: network={network_time*1000:.1f}ms, compute={compute_time*1000:.1f}ms, total_latency={latency*1000:.1f}ms")
        
        # Merge timing from result
        result_timing = result.get("timing", {})
        if isinstance(result_timing, dict):
            result_timing.update(timing_dict)
        
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
                "timing": result_timing,
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

        # Create timing dictionary to collect all timing metrics
        timing_dict: Dict[str, float] = {}

        # Time image bytes retrieval
        image_bytes_start = perf_counter()
        image_bytes = ctx.image_bytes()
        if image_bytes is None:
            return None
        image_bytes_duration = perf_counter() - image_bytes_start
        timing_dict["image_bytes_retrieve_s"] = image_bytes_duration

        # Time image encoding
        image_b64 = encode_image_for_transport(image_bytes, timing_dict=timing_dict)

        metadata = {
            "timestamp": ctx.metadata.get("timestamp"),
            "camera_id": ctx.metadata.get("camera_id"),
            "video_file": ctx.metadata.get("video_file"),
            "video_timestamp_sec": ctx.metadata.get("video_timestamp_sec"),
        }
        result = self.remote_backend.run_llm(
            image_b64=image_b64,
            frame_index=frame_index,
            fsm_state=state,
            metadata=metadata,
            sensor_data=ctx.sensor_data,
            sensor_baseline=ctx.metadata.get("sensor_baseline"),
            sensor_anomalies=ctx.metadata.get("sensor_anomalies"),
            timing_dict=timing_dict,
        )
        
        # Merge timing from result
        if isinstance(result, dict):
            result_timing = result.get("timing", {})
            if isinstance(result_timing, dict):
                result_timing.update(timing_dict)
            elif not result_timing:
                result["timing"] = timing_dict
        
        return result
