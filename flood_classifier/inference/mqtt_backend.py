"""MQTT utilities for synchronous request/response style inference offloading.

This module provides :class:`MQTTInferenceClient` which allows the Pi to publish
an inference job and wait for the Jetson's reply using correlation IDs.  It
assumes the Jetson publishes heartbeat messages so we can quickly detect
connectivity problems and fall back to local inference.
"""
from __future__ import annotations

import base64
import json
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Optional

import config
from time import perf_counter

try:  # pragma: no cover - optional dependency for remote inference
    import paho.mqtt.client as mqtt
except Exception:  # pragma: no cover - allow unit tests without paho-mqtt
    mqtt = None  # type: ignore


@dataclass
class MQTTResponse:
    """Container storing a Jetson response payload and metadata."""

    data: Dict[str, object]
    latency_s: float
    sent_at: float
    received_at: float


class MQTTInferenceClient:
    """Publish inference requests to the Jetson and wait for replies."""

    def __init__(
        self,
        client_id: Optional[str] = None,
        request_topic: str = config.MQTT_INFERENCE_REQUEST_TOPIC,
        response_topic: str = config.MQTT_INFERENCE_RESPONSE_TOPIC,
        heartbeat_topic: str = config.MQTT_INFERENCE_HEARTBEAT_TOPIC,
        qos: int = config.MQTT_INFERENCE_QOS,
        timeout_s: float = config.MQTT_INFERENCE_TIMEOUT,
    ) -> None:
        if mqtt is None:
            raise RuntimeError(
                "paho-mqtt is not installed; cannot use MQTTInferenceClient"
            )

        self.request_topic = request_topic.rstrip("/")
        self.response_topic = response_topic.rstrip("/")
        self.heartbeat_topic = heartbeat_topic
        self.qos = qos
        self.timeout_s = timeout_s

        cid = client_id or f"pi-inference-{uuid.uuid4().hex[:8]}"
        self._client = mqtt.Client(client_id=cid)
        self._client.enable_logger()

        self._pending: Dict[str, threading.Event] = {}
        self._payloads: Dict[str, MQTTResponse] = {}
        self._sent_at: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._connected = threading.Event()
        self._last_heartbeat = 0.0

        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.on_disconnect = self._on_disconnect

        print(f"🔍 [MQTTBackend] Connecting to broker at {config.MQTT_BROKER_URL}:{config.MQTT_BROKER_PORT}...")
        self._client.connect(config.MQTT_BROKER_URL, config.MQTT_BROKER_PORT)
        self._client.loop_start()

        if not self._connected.wait(timeout=5.0):
            print(f"❌ [MQTTBackend] Failed to connect to broker within 5 seconds")
            raise RuntimeError("MQTTInferenceClient failed to connect to broker")

        print(f"✅ [MQTTBackend] Connected to broker successfully")
        self._client.subscribe(f"{self.response_topic}/#", qos=self.qos)
        self._client.subscribe(self.heartbeat_topic, qos=0)
        print(f"🔍 [MQTTBackend] Subscribed to response topic '{self.response_topic}/#' and heartbeat topic '{self.heartbeat_topic}'")

    # ------------------------------------------------------------------
    def _on_connect(self, _client, _userdata, _flags, rc):
        if rc == 0:
            print(f"✅ [MQTTBackend] MQTT connection established (rc={rc})")
            self._connected.set()
        else:
            print(f"❌ [MQTTBackend] MQTT connection failed (rc={rc})")

    # ------------------------------------------------------------------
    def _on_disconnect(self, _client, _userdata, _rc):
        self._connected.clear()

    # ------------------------------------------------------------------
    def _on_message(self, _client, _userdata, msg):
        topic = msg.topic
        if topic == self.heartbeat_topic:
            self._last_heartbeat = time.time()
            print(f"💓 [MQTTBackend] Received heartbeat from Jetson at {time.strftime('%H:%M:%S')}")
            return

        if not topic.startswith(self.response_topic):
            return

        # Time JSON deserialization
        json_deserialize_start = perf_counter()
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            json_deserialize_duration = perf_counter() - json_deserialize_start
        except Exception:
            return

        correlation_id = payload.get("id")
        if not correlation_id:
            return

        with self._lock:
            event = self._pending.get(correlation_id)
            if event is None:
                return
            received_at = time.time()
            sent_at = self._sent_at.get(correlation_id, received_at)
            latency = received_at - sent_at
            # Store JSON deserialize timing in response data if available
            if isinstance(payload, dict) and "timing" in payload:
                payload["timing"]["mqtt_json_deserialize_s"] = json_deserialize_duration
            self._payloads[correlation_id] = MQTTResponse(payload, latency, sent_at, received_at)
            event.set()

    # ------------------------------------------------------------------
    def publish_request(self, message: Dict[str, object], timing_dict: Optional[Dict[str, float]] = None) -> MQTTResponse:
        """Publish a request and block until the response arrives or times out."""

        correlation_id = message.setdefault("id", uuid.uuid4().hex)
        message["ts"] = time.time()

        event = threading.Event()
        with self._lock:
            self._pending[correlation_id] = event

        try:
            # Time JSON serialization
            json_serialize_start = perf_counter()
            payload = json.dumps(message).encode("utf-8")
            json_serialize_duration = perf_counter() - json_serialize_start
            if timing_dict is not None:
                timing_dict["mqtt_json_serialize_s"] = json_serialize_duration

            if len(payload) > config.MQTT_INFERENCE_MAX_PAYLOAD:
                raise ValueError("Inference payload exceeds MQTT_INFERENCE_MAX_PAYLOAD")

            # Time MQTT publish
            mqtt_publish_start = perf_counter()
            result = self._client.publish(self.request_topic, payload, qos=self.qos)
            mqtt_publish_duration = perf_counter() - mqtt_publish_start
            if timing_dict is not None:
                timing_dict["mqtt_publish_s"] = mqtt_publish_duration

            # Time wait for publish confirmation
            wait_publish_start = perf_counter()
            result.wait_for_publish(timeout=self.timeout_s)
            wait_publish_duration = perf_counter() - wait_publish_start
            if timing_dict is not None:
                timing_dict["mqtt_wait_publish_s"] = wait_publish_duration

            with self._lock:
                self._sent_at[correlation_id] = time.time()

            # Time waiting for response
            wait_response_start = perf_counter()
            if not event.wait(timeout=self.timeout_s):
                wait_response_duration = perf_counter() - wait_response_start
                if timing_dict is not None:
                    timing_dict["mqtt_wait_response_s"] = wait_response_duration
                raise TimeoutError("Timed out waiting for Jetson inference response")
            wait_response_duration = perf_counter() - wait_response_start
            if timing_dict is not None:
                timing_dict["mqtt_wait_response_s"] = wait_response_duration

            with self._lock:
                resp = self._payloads.pop(correlation_id)
                self._pending.pop(correlation_id, None)
                self._sent_at.pop(correlation_id, None)
                return resp

        finally:
            with self._lock:
                self._pending.pop(correlation_id, None)
                self._payloads.pop(correlation_id, None)
                self._sent_at.pop(correlation_id, None)

    # ------------------------------------------------------------------
    def is_healthy(self, heartbeat_timeout: float = 10.0) -> bool:
        connected = self._connected.is_set()
        last_heartbeat = self._last_heartbeat
        time_since_heartbeat = time.time() - last_heartbeat if last_heartbeat > 0.0 else float('inf')
        is_healthy_result = connected and last_heartbeat > 0.0 and time_since_heartbeat <= heartbeat_timeout
        
        if not is_healthy_result:
            print(f"🔍 [MQTTBackend] is_healthy: connected={connected}, last_heartbeat={last_heartbeat}, "
                  f"time_since_heartbeat={time_since_heartbeat:.2f}s, timeout={heartbeat_timeout}s, "
                  f"healthy={is_healthy_result}")
        
        return is_healthy_result

    # ------------------------------------------------------------------
    def close(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()


def encode_image_for_transport(image_bytes: bytes, timing_dict: Optional[Dict[str, float]] = None) -> str:
    """Return a compact base64 representation suitable for MQTT."""
    encode_start = perf_counter()
    result = base64.b64encode(image_bytes).decode("utf-8")
    encode_duration = perf_counter() - encode_start
    if timing_dict is not None:
        timing_dict["image_encode_s"] = encode_duration
    return result
